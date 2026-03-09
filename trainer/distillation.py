# Adopted from https://github.com/guandeh17/Self-Forcing  # 采用自 Self-Forcing 开源项目
# SPDX-License-Identifier: Apache-2.0  # 采用 Apache 2.0 开源协议
import gc  # 引入 python 的垃圾回收模块，用于手动清理内存
import logging  # 引入日志模块，用于输出和记录运行日志
import random  # 引入随机数模块，用于生成随机种子等
import re  # 引入正则表达式模块
from pathlib import Path  # 从 pathlib 引入 Path，用于方便地处理文件路径

from utils.dataset import TextDataset, TwoTextDataset, cycle, ShardingLMDBDataset  # 导入自定义的数据集类和 utils
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job  # 导入分布式训练相关的工具 (FSDP 和 EMA)
from utils.misc import (  # 导入杂项工具函数
    set_seed,  # 设置随机种子的函数，保证可复现性
    merge_dict_list  # 字典列表合并函数
)
import torch.distributed as dist  # 导入 PyTorch 的分布式计算模块
from omegaconf import OmegaConf  # 导入 OmegaConf，用于解析和管理 yaml 配置
from model import DMD, DMDSwitch  # 导入 DMD 和 DMDSwitch 模型架构
from model.streaming_training import StreamingTrainingModel  # 导入支持流式训练的模型架构
import torch  # 导入 PyTorch 核心库
import wandb  # 导入 Weights & Biases 库，用于实验追踪和可视化
import time  # 导入时间模块，用于计算单步耗时等
import os  # 导入系统操作系统接口模块，用于环境变量和路径操作
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入完全切片数据并行 (FSDP)，用于大规模模型训练
from torch.distributed.fsdp import (  # 导入 FSDP 相关的状态字典配置枚举和类
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig
)
from torchvision.io import write_video  # 导入 torchvision_io 的视频写入功能

# LoRA related imports  # LoRA 相关的库导入
import peft  # 导入 PEFT 库，用于参数高效微调 (如 LoRA)
from peft import get_peft_model_state_dict  # 导入获取 PEFT 模型状态字典的方法
import safetensors.torch  # 导入 safetensors 库，用于安全快速的模型张量存取

from utils.memory import gpu, get_cuda_free_memory_gb, log_gpu_memory  # 导入 GPU 显存监控和日志分析工具
from pipeline import (  # 导入推理 Pipeline
    CausalInferencePipeline,  # 因果推理/自回归推理 Pipeline
    SwitchCausalInferencePipeline  # 带切换机制的推理 Pipeline
)
from utils.debug_option import DEBUG, LOG_GPU_MEMORY, DEBUG_GRADIENT  # 导入各种 debug 标志位
try:
    from one_logger_utils import OneLoggerUtils  # 尝试导入内部的 OneLogger 监控工具
except ImportError:
    OneLoggerUtils = None  # 如果没有该工具包则置为 None
import time  # 再次导入 time (有重复)

class Trainer:  # 定义训练器基类
    
    def __init__(self, config):  # Trainer 初始化函数，接收配置对象
        self.config = config  # 保存配置对象到实例属性
        self.step = 0  # 初始化当前训练步数为 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)  # 步骤 1: 初始化分布式训练环境（进程号、种子、数据类型、日志等）
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许 CUDA 的矩阵乘法使用 TF32 加速计算
        torch.backends.cudnn.allow_tf32 = True  # 允许 cuDNN 使用 TF32 加速计算

        launch_distributed_job()  # 启动分布式任务（设置环境变量等）
        global_rank = dist.get_rank()  # 获取当前进程的全局 Rank（进程序号）
        self.world_size = dist.get_world_size()  # 获取参与训练的总进程数 (GPU 数量)

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32  # 如果开启混合精度，数据类型设为 bf16，否则使用 fp32
        self.device = torch.cuda.current_device()  # 获取当前进程绑定的 CUDA 设备 ID
        self.is_main_process = global_rank == 0  # 判断当前进程是否为主进程 (Rank 0)
        self.causal = config.causal  # 读取配置中是否使用因果 (causal) 模式的标志
        self.disable_wandb = config.disable_wandb  # 读取配置中是否禁用 weights & biases

        # use a random seed for the training  # 为训练设置随机种子
        if config.seed == 0:  # 如果配置中的种子为 0，则生成一个真正的随机种子
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)  # 在设备上生成一个随机数
            dist.broadcast(random_seed, src=0)  # 主进程将生成的随机种子广播给所有其他进程，确保大家用相同的种子基数
            config.seed = random_seed.item()  # 更新 config.seed 的值

        set_seed(config.seed + global_rank)  # 在基数上加上当前进程的 Rank，确保不同 GPU 的随机性有差异，但又是可复现的

        self.use_one_logger = getattr(config, "use_one_logger", True)  # 读取配置中是否使用 one_logger，默认为 True
        if self.is_main_process and not self.disable_wandb:  # 如果是主进程且没有禁用 wandb
            wandb.login(  # 登录 wandb 账号
                # host=config.wandb_host,
                key=config.wandb_key)  # 使用配置中的 api key 进行登录
            wandb.init(  # 初始化 wandb 项目
                config=OmegaConf.to_container(config, resolve=True),  # 记录当前训练的所有配置参数 (转化为基础字典)
                name=config.config_name,  # 设置 wandb 运行名称
                mode="online",  # 设置模式为在线同步
                entity=config.wandb_entity,  # 设置 wandb 团队/用户名
                project=config.wandb_project,  # 设置 wandb 项目名
                dir=config.wandb_save_dir  # 设置 wandb 日志保存的本地目录
            )

        self.output_path = config.logdir  # 获取训练输出和 checkpoint 的保存路径
        app_start_time = time.time_ns() / 1_000_000  # 获取当前系统时间的毫秒级时间戳，用于记录 app 启动时间
        
        # ------------------------------------- One Logger Setup ----------------------------------------------  # One Logger (指标上报组件) 设置
        if self.use_one_logger and dist.get_rank() == 0 and not self.disable_wandb:  # 如果使用该 logger 且是主进程且开启了日志上报
            app_tag_run_name = f"dmd_{config.real_name[:6]}_local_attn_size_{config.model_kwargs.local_attn_size}_lr_{config.lr}"  # 组装基于模型和参数的运行名称tag
            app_tag_run_version = "0.0.0"  # 设置运行版本号
            app_tag = f"{app_tag_run_name}_{app_tag_run_version}_{config.batch_size}_{dist.get_world_size()}"  # 结合 batch_size 和 显卡数 拼装最终的 app_tag
            one_logger_config = {  # 组装 one logger 的配置字典
                "enable_for_current_rank": True,  # 允许当前rank上报
                "one_logger_async": True,  # 开启异步上报以防阻塞训练
                "one_logger_project": getattr(config, "one_logger_project", "self-forcing"),  # 设置项目名为 self-forcing
                "log_every_n_train_iterations": getattr(config, "log_iters", 10),  # 每隔 N 步进行一次日志上报
                "app_tag_run_version": app_tag_run_version,  # 版本信息
                "summary_data_schema_version": "1.0.0",  # schema 版本
                "app_run_type": "training",  # 设置上报任务的类型为 training
                "app_tag": app_tag,  # 任务标签
                "app_tag_run_name": app_tag_run_name,  # 运行名字
                "one_logger_run_name": app_tag_run_name,  # logger 的运行名字
                "world_size": dist.get_world_size(),  # 节点/显卡总数
                "global_batch_size": config.batch_size * getattr(config, "gradient_accumulation_steps", 1) * dist.get_world_size(),  # 计算全局等效的 batch size
                "batch_size": config.batch_size,  # 单卡 batch size
                "train_iterations_target": getattr(config, "max_iters", 0),  # 设置最大训练步数目标
                "train_samples_target": (getattr(config, "max_iters", 0) * config.batch_size) if getattr(config, "max_iters", 0) else 0, # 设置最大训练样本数目标
                "is_train_iterations_enabled": True,  # 标记开启训练迭代上报
                "is_baseline_run": False,  # 标记这不是作为 baseline
                "is_test_iterations_enabled": False,  # 标记不开启测试迭代上报
                "is_validation_iterations_enabled": True,  # 标记开启验证迭代上报
                "is_save_checkpoint_enabled": True,  # 标记关注 checkpoint 保存的事件
                "is_log_throughput_enabled": False,  # 标记不专门开启吞吐量独立上报
                "micro_batch_size": config.batch_size,  # 设置 micro batch size
                "seq_length": getattr(config, "image_or_video_shape")[1] * getattr(config, "image_or_video_shape")[3] * getattr(config, "image_or_video_shape")[4],  # 估算并设置一个粗略的序列长度 (其实这里计算不准确)
                "save_checkpoint_strategy": "sync",  # 指定是以同步模式保存 checkpoint
            }
            self.one_logger = OneLoggerUtils(one_logger_config)  # 根据组装好的配置实例化 logger 工具
            self.one_logger.on_app_start(app_start_time = app_start_time)  # 触发 app 启动事件
        else:
            self.one_logger = None  # 否则不使用该 logger

        # Step 2: Initialize the model  # 步骤 2: 初始化计算模型
        if self.one_logger is not None:  # 如果开启了上报
            self.one_logger.on_model_init_start()  # 触发模型初始化开始事件

        if config.distribution_loss == "causvid":  # 如果使用 causvid 分布匹配算法
            self.model = CausVid(config, device=self.device)  # 实例化 CausVid 模型
        elif config.distribution_loss == "dmd":  # 如果使用 dmd 算法
            self.model = DMD(config, device=self.device)  # 实例化 DMD 模型
        elif config.distribution_loss == "dmd_switch":  # 如果使用 dmd_switch (可能支持模式切换的变体)
            self.model = DMDSwitch(config, device=self.device)  # 实例化带切换机制的 DMD 模型
        elif config.distribution_loss == "dmd_window":  # 如果使用基于局部窗口的 dmd
            self.model = DMDWindow(config, device=self.device)  # 实例化基于 Window 的 DMD 模型
        elif config.distribution_loss == "sid":  # 如果使用 sid 算法
            self.model = SiD(config, device=self.device)  # 实例化 SiD 模型
        else:
            raise ValueError("Invalid distribution matching loss")  # 若均不符合则抛出异常

        # Save pretrained model state_dicts to CPU  # 将判别器（teacher）预训练权重的 state_dict 暂存到 CPU 内存
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()  # 获取初始的预训练权重字典用于对照或后续备份

        # Auto resume configuration (needed for LoRA checkpoint loading)  # 自动恢复训练的配置（同时也用于决定加载哪个 LoRA 断点）
        auto_resume = getattr(config, "auto_resume", True)  # Default to True  # 默认开启 auto_resume 机制

        # ================================= LoRA Configuration =================================  # 设置 LoRA 组件
        self.is_lora_enabled = False  # 初始化标识变量为 False
        self.lora_config = None  # 初始化配置为空
        if hasattr(config, 'adapter') and config.adapter is not None:  # 如果提供了 adapter (LoRA) 配置
            self.is_lora_enabled = True  # 开启 LoRA 模式
            self.lora_config = config.adapter  # 读取 LoRA 的具体配置字典
            
            if self.is_main_process:  # 在主进程打日志
                print(f"LoRA enabled with config: {self.lora_config}")  # 打印 LoRA 配置
                print("Loading base model and applying LoRA before FSDP wrapping...")  # 提示先应用 LoRA 后再套 FSDP 引擎
            
            # 1. Load base model first (config.generator_ckpt) - before applying LoRA and FSDP  # 1. 优先加载基础主干权重 (然后再注 LoRA、包 FSDP)
            base_checkpoint_path = getattr(config, "generator_ckpt", None)  # 获取基础生成器的模型路径
            if base_checkpoint_path:  # 如果存在
                if self.is_main_process:
                    print(f"Loading base model from {base_checkpoint_path} (before applying LoRA)")  # 打印开始加载底模
                base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu")  # 因为内存/显存限制，先把模型字典加载进 CPU
                
                # Load generator (directly; no key alignment needed since LoRA not applied yet)  # 首先加载 Generator (基础非 LoRA 层权重)
                if "generator" in base_checkpoint:  # 判断字典内是否包了一层 "generator"
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {base_checkpoint_path}")
                    result = self.model.generator.load_state_dict(base_checkpoint["generator"], strict=True)  # 用严格模式匹配所有的键并赋值
                    if self.is_main_process:
                        print("Generator weights loaded successfully")
                elif "model" in base_checkpoint:  # 如果直接包含 "model" 层
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {base_checkpoint_path}")
                    result = self.model.generator.load_state_dict(base_checkpoint["model"], strict=True)  # 解包并加载
                    if self.is_main_process:
                        print("Generator weights loaded successfully")
                else:  # 若前缀都没匹配上
                    if self.is_main_process:
                        print("Warning: Generator checkpoint not found in base model.")  # 抛出警告
                
                # Load critic  # 尝试同时加载 Critic 基础模型 (判别器/目标分布导师模型)
                if "critic" in base_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained critic from {base_checkpoint_path}")
                    result = self.model.fake_score.load_state_dict(base_checkpoint["critic"], strict=True)  # 加载 Critic 主体模型权重
                    if self.is_main_process:
                        print("Critic weights loaded successfully")
                else:
                    if self.is_main_process:
                        print("Warning: Critic checkpoint not found in base model.")
            else:  # 若根本找不到基础模型参数文件，则在 LoRA 场景下直接报错，因为必须有好的底模基石
                if self.is_main_process:
                    raise ValueError("No base model checkpoint specified for LoRA training.")
            
            # Load training step  # 加载基础模型中的 steps 计数
            if "step" in base_checkpoint:
                self.step = base_checkpoint["step"]  # 继承训练轮数计数器
                if self.is_main_process:
                    print(f"base_checkpoint step: {self.step}")
            else:
                if self.is_main_process:
                    print("Warning: Step not found in checkpoint, starting from step 0.")  # 没有就从0开始
            
            # 2. Apply LoRA wrapping now (after loading base model, before FSDP wrapping)  # 2. 调用工具函数在网络相应线性层外层包装 LoRA 适配器
            if self.is_main_process:
                print("Applying LoRA to models...")
            self.model.generator.model = self._configure_lora_for_model(self.model.generator.model, "generator")  # 对 Generator 添加 lora 层
            
            # Configure LoRA for fake_score if needed  # 根据配置决定是否也要对 Critic (假分数评估者) 加入 LoRA 参数
            if getattr(self.lora_config, 'apply_to_critic', True):
                self.model.fake_score.model = self._configure_lora_for_model(self.model.fake_score.model, "fake_score")  # 封装 LoRA
                if self.is_main_process:
                    print("LoRA applied to both generator and critic")
            else:
                if self.is_main_process:
                    print("LoRA applied to generator only")
            
            # 3. Load LoRA weights before FSDP wrapping (if a checkpoint is available)  # 3. 如果是自动恢复续训，则在上 FSDP 框架前注入 LoRA 保存参数
            lora_checkpoint_path = None
            if auto_resume and self.output_path:  # 如果开启自动续约，且定义了输出文件夹
                # Find the latest checkpoint and verify it is a LoRA checkpoint  # 在目录下寻找最新 step 的保存文件夹
                latest_checkpoint = self.find_latest_checkpoint(self.output_path)
                if latest_checkpoint:  # 如果找到了已保存的 checkpoint
                    try:
                        checkpoint = torch.load(latest_checkpoint, map_location="cpu")  # 将断点文件读入内存
                        if "generator_lora" in checkpoint and "critic_lora" in checkpoint:  # 确保这个是包含了 LoRA 参数的合法断点
                            lora_checkpoint_path = latest_checkpoint
                            if self.is_main_process:
                                print(f"Auto resume: Found LoRA checkpoint at {lora_checkpoint_path}")
                        else:  # 否则断点不合法，可能是之前全量微调保存下来的
                            raise ValueError(f"Checkpoint {latest_checkpoint} is not a LoRA checkpoint. "
                                           f"Found keys: {list(checkpoint.keys())}")
                    except Exception as e:
                        if self.is_main_process:
                            print(f"Error validating checkpoint: {e}")  # 报告解析断点遇到错误
                        raise e  # 将错误向上抛出
                else:  # 没有找到最新断点
                    if self.is_main_process:
                        print("Auto resume: No LoRA checkpoint found in logdir")
            elif auto_resume:  # 没配置 logdir 虽然开启了自动续约，也无法运行
                if self.is_main_process:
                    print("Auto resume enabled but no logdir specified for LoRA")
            else:  # 主动禁用了自动续约机制
                if self.is_main_process:
                    print("Auto resume disabled for LoRA")
            
            # If no auto-resumed LoRA checkpoint found, try config.lora_ckpt  # 如果按自动恢复机制没找到特定 LoRA 断点，尝试读取用户指定的固定 lora 权重路径
            if lora_checkpoint_path is None:  # 如果仍未确定要加载的具体 LoRA 断点
                lora_ckpt_path = getattr(config, "lora_ckpt", None)  # 获取 config 中标明的 lora 权重路径
                if lora_ckpt_path:  # 如果提供了该固定路径
                    try:
                        checkpoint = torch.load(lora_ckpt_path, map_location="cpu")  # 将文件加载至 CPU
                        if "generator_lora" in checkpoint and "critic_lora" in checkpoint:  # 验证是否包含预期的 lora 字段
                            lora_checkpoint_path = lora_ckpt_path  # 确定路径为将要加载的这一合法路径
                            if self.is_main_process:
                                print(f"Using explicit LoRA checkpoint: {lora_checkpoint_path}")  # 报告使用了显式声明的权重
                        else:  # 字段不满足，抛出警告
                            raise ValueError(f"Explicit LoRA checkpoint {lora_ckpt_path} is not a valid LoRA checkpoint. "
                                           f"Found keys: {list(checkpoint.keys())}")  # 列出其中实际存在的 key 方便排查
                    except Exception as e:
                        if self.is_main_process:
                            print(f"Error loading explicit LoRA checkpoint: {e}")  # 捕获并打印读取/解析该显式断点产生的异常
                        raise e  # 将错误向上抛出结束程序
                else:  # 若也没有显式指定，说明是刚开始训练
                    if self.is_main_process:
                        print("No LoRA checkpoint specified, starting LoRA training from scratch")  # 提醒是从头开始重新训练 LoRA
            
            # Load LoRA checkpoint (before FSDP wrapping)  # 实际开始把刚确定好的 LoRA 断点内容填充到已包装了适配器的网络中
            if lora_checkpoint_path:
                if self.is_main_process:
                    print(f"Loading LoRA checkpoint from {lora_checkpoint_path} (before FSDP wrapping)")
                lora_checkpoint = torch.load(lora_checkpoint_path, map_location="cpu")  # 执行加载 (若前面已加载这里有点小重复，但不影响)
                
                # Load LoRA weights using PEFT's standard method  # 使用由 PEFT 开源库提供的标准方法加载权重
                if "generator_lora" in lora_checkpoint:
                    if self.is_main_process:
                        print(f"Loading LoRA generator weights: {len(lora_checkpoint['generator_lora'])} keys in checkpoint")
                    
                    # Use PEFT's set_peft_model_state_dict; it automatically aligns key names  # 利用内置 API 注入，它会自动帮我们对齐键名
                    peft.set_peft_model_state_dict(self.model.generator.model, lora_checkpoint["generator_lora"])
                
                if "critic_lora" in lora_checkpoint:  # 如果包含判别器的 LoRA 回调
                    if self.is_main_process:
                        print(f"Loading LoRA critic weights: {len(lora_checkpoint['critic_lora'])} keys in checkpoint")
                    
                    # Use PEFT's set_peft_model_state_dict; it automatically aligns key names  # 同样对齐并注入判别器中
                    peft.set_peft_model_state_dict(self.model.fake_score.model, lora_checkpoint["critic_lora"])

                # Load training step  # 读取当前断点处的轮次 step
                if "step" in lora_checkpoint:
                    self.step = lora_checkpoint["step"]  # 赋值给自身迭代步数
                    if self.is_main_process:
                        print(f"Resuming LoRA training from step {self.step}")  # 提示从中继续
            else:
                if self.is_main_process:
                    print("No LoRA checkpoint to load, starting from scratch")  # 无断点，说明完全从零训练

        self.model.generator = fsdp_wrap(  # 使用自定义的 fsdp_wrap 封装类将被包裹了 LoRA/或者保持原貌的基础生成器用 FSDP 初始化并行环境
            self.model.generator,  # 传入实际模型模块
            sharding_strategy=config.sharding_strategy,  # 全量切片、参数切片还是其他并行策略
            mixed_precision=config.mixed_precision,  # 是否开启混合精度训练 bf16
            wrap_strategy=config.generator_fsdp_wrap_strategy  # 提供在包裹嵌套层时所要使用的特定分解策略规则（例如按层大小 wrap）
        )

        self.model.real_score = fsdp_wrap(  # 包装真实图像分布老师模型
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(  # 包装辨别模型自己生成分布(生成器所受评估)的模型，注意这是被 Generator 更新以及自己需要去噪优化的核心模块
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        # FSDP wrap image_encoder for I2V training  # 为图像到视频 (I2V) 任务专门使用 FSDP 包装 image encoder 编码器
        if self.config.i2v:  # 检查是否是在 I2V 模式下
            self.model.image_encoder = fsdp_wrap(  # 包装 Image Encoder
                self.model.image_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=getattr(config, "image_encoder_fsdp_wrap_strategy", "size"),  # 按网络层尺寸切片
                min_num_params=int(5e6),  # 参数大于 500 万时就进行一次包装
                cpu_offload=getattr(config, "image_encoder_cpu_offload", False)  # 是否将不活跃参数从显存倒出给 CPU 以节省现存
            )
            self.model.vae = self.model.vae.to(  # 因为是 I2V ，VAE 可能需要对输入图像作编码，将它放入相应训练用设备及精度
                device=self.device, dtype=torch.bfloat16)  # 移至 GPU 与 bf16 精度
        else:  # 如果是文本到视频 (T2V)
            self.model.vae = self.model.vae.to(  # 基础加载 VAE
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.model.text_encoder = fsdp_wrap(  # 同样地，把 Text Encoder (如 CLIP，T5) 包装进 FSDP 控制循环
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)  # 是否将 Text Encoder 参数暂存 CPU （非常有利于大模型节约内存）
        )

        # if not config.no_visualize or config.load_raw_video:
        #     print("Moving vae to device 2, self.device: ", self.device)
        #     self.model.vae = self.model.vae.to(
        #         device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # Step 3: Set up EMA parameter containers  # 步骤 3: 建立并设置指数移动平均 (EMA) 的权重容器以获得更鲁棒的合成评估
        rename_param = (  # 这一个 lambda 函数为了清洗参数名，将 FSDP 添加的前缀后缀抹去以对齐
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}  # 保存全部需要开启梯度的参数名引用
        for n, p in self.model.generator.named_parameters():  # 遍历生成器的每个参数
            if not p.requires_grad:  # 冻结的参数不用处理
                continue

            renamed_n = rename_param(n)  # 剥去包装前缀获取纯净的参数名
            self.name_to_trainable_params[renamed_n] = p  # 存储在参数字典备用（后续 EMA 等会用）
        ema_weight = config.ema_weight  # 获取 EMA 滑动平均的衰减因子
        self.generator_ema = None  # 初始设 EMA 容器为空
        if (ema_weight is not None) and (ema_weight > 0.0):  # 检查是否设定启用 EMA
            if self.is_lora_enabled:  # 但要注意，如果我们在用 LoRA 高效微调，那通常不建立 EMA
                if self.is_main_process:
                    print(f"EMA disabled in LoRA mode (LoRA provides efficient parameter updates without EMA)")  # 提供明确提示，因 LoRA 本身就足够稳健且为了效率避免翻倍内存占用
                self.generator_ema = None
            else:  # 若是全量微调，应当启动 EMA
                print(f"Setting up EMA with weight {ema_weight}")
                self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)  # 用自定义 FSDP 支持的 EMA 模块包装生成器

        
        if self.one_logger is not None:
            self.one_logger.on_model_init_end()  # 结束监控模型初始化上报节点
        
        # Step 4: Initialize the optimizer  # 步骤 4: 实例化所需优化器结构
        if self.one_logger is not None:
            self.one_logger.on_optimizer_init_start()  # 开始初监控优化器初始化

        self.generator_optimizer = torch.optim.AdamW(  # 使用 AdamW 算法来实例化 Generator 优化器
            [param for param in self.model.generator.parameters()  # 所有 generator 中带有 requires_grad = True 的可微参数都会被优化
             if param.requires_grad],
            lr=config.lr,  # 学习率
            betas=(config.beta1, config.beta2),  # adamw 冲量算法内部 betas 系数
            weight_decay=config.weight_decay  # 衰减率惩罚因数 L2 正则
        )

        self.critic_optimizer = torch.optim.AdamW(  # 判别器也需要一个独立分离出来的优化器
            [param for param in self.model.fake_score.parameters()  # 由于 critic 会进行自身去噪学习
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,  # 解析其独立专属学习率，通常不同于 generator 并更大些获得足够的引导反馈
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        if self.one_logger is not None:
            self.one_logger.on_optimizer_init_end()   # 结束优化器节点上报

        # Step 5: Initialize the dataloader  # 步骤 5: 配置以及初始化数据集和 DataLoader 数据装载器
        if self.one_logger is not None:
            self.one_logger.on_dataloader_init_start()  # 开始数据引擎初始化监控
        if self.config.i2v:  # 在 I2V (图生视频) 特殊场景下
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))  # 选取包含 Sharding 切片功能的超大数据级 LMDB 图文支持结构集
        elif self.config.distribution_loss == "dmd_switch":  # 对于需要在不同长短句引导分布中来回跳跃切换（Switch）的 DMD 方法下
            dataset = TwoTextDataset(config.data_path, config.switch_prompt_path)  # 读取双重对照的数据
        else:  # 正常状态的 T2V 或者单纯 DMD
            dataset = TextDataset(config.data_path)  # 选用普通基于纯文本读取集的 dataset 读取法
        sampler = torch.utils.data.distributed.DistributedSampler(  # 利用基于分布式下的随机并行均匀分配重分区的采样器
            dataset, shuffle=True, drop_last=True)  # 需要按 batch 切除零头使得分布均衡并防止报错，同时打开数据彻底洗牌
        dataloader = torch.utils.data.DataLoader(  # 把带有并行取样机制打包的数据源投入到正式工作引擎器里
            dataset,
            batch_size=config.batch_size,  # 小批量的配置 size 大小，依据机器显存所限而决定
            sampler=sampler,  # 挂载 DDP 的并行随机选择调度器
            num_workers=8)  # 并用设置工作进程（CPU核心使用数量）多线程 IO 拉取速度加倍

        if dist.get_rank() == 0:  # 为了不引起控制台刷屏，打印出有几条数据只能是 node 0 去做
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)  # 无限循环产生器使得数据如果消耗见底会通过 iterator next () 不断首尾重连不断读取循环不结束

        # Step 6: Initialize the validation dataloader for visualization (fixed prompts)  # 步骤 6：准备用于可视化和评估的数据引擎
        self.fixed_vis_batch = None  # 置一个属性槽装载确定的一个特定测试图片与短提示的数据槽
        self.vis_interval = getattr(config, "vis_interval", -1)  # 多久或者运行迭代几次做一次可视抽查测出，配置不在默认为 -1 永不测
        if self.vis_interval > 0 and len(getattr(config, "vis_video_lengths", [])) > 0:  # 若启用了定期的自验并确定了我们要看结果的是怎么的一个视频长短配置（比如有 8, 16..21)
            # Determine validation data path  # 决定好要用配置内的特定检验用的 validation test 数据集路径，没用单独的就是采用之前主训同一条
            val_data_path = getattr(config, "val_data_path", None) or config.data_path

            if self.config.i2v:  # 基于同样的对应规则将评估用的数据实体以同样种类类名作一次创建实例化过程
                val_dataset = ShardingLMDBDataset(val_data_path, max_pair=int(1e8))
            elif self.config.distribution_loss == "dmd_switch":
                val_dataset = TwoTextDataset(val_data_path, config.val_switch_prompt_path)
            else:
                val_dataset = TextDataset(val_data_path)

            if dist.get_rank() == 0:
                print("VAL DATASET SIZE %d" % len(val_dataset))  # 主显卡上将测试集合长数量进行一次终端显露回馈

            sampler = torch.utils.data.distributed.DistributedSampler(  # 给这批特测集附带分派也一样附带 sampler ，只是要重点说明关闭彻底随机而是顺序提取
                val_dataset, shuffle=False, drop_last=False)  # 验证时候需要稳定重复地比对比率以及不放弃任意边界碎渣的批流数据保留
            # streaming sampling to keep prompts fixed  # 为了长期维持稳定对照不改变对象，抽提方式必须按先后定顺序取
            val_dataloader = torch.utils.data.DataLoader(  # 构建评估环节工作线程库以及引擎启动拉送的管线管道容器
                val_dataset,
                batch_size=getattr(config, "val_batch_size", 1),  # 一般在作验的时候都是将单个任务放于 GPU 进行独门观察确保准确不会冲突混合，批取常为了节省时间
                sampler=sampler,
                num_workers=8,
            )

            # Take the first batch as fixed visualization batch  # 单取最头部的这一个小单元子集存放进入内存长期用做评估时提供相同的原始物料做公平参考系
            try:
                self.fixed_vis_batch = next(iter(val_dataloader))  # 把该数据结构装载成为固定迭代并把该头一个给抽出备齐存放起来
            except StopIteration:
                self.fixed_vis_batch = None  # 防止空库造成的出错抛出，给予捕捞后的一个无数据宽容
            
            # ----------------------------------------------------------------------------------------------------------
            # Visualization settings  # 图形化的生成视频输出配置管理段落
            # ----------------------------------------------------------------------------------------------------------
            # List of video lengths to visualize, e.g. [8, 16, 32]  # 有些时候我们需要同时对照生成多短中长不一的视觉内容产出的差异，比如 8 或 16 的长度列表等
            self.vis_video_lengths = getattr(config, "vis_video_lengths", [])

            if self.vis_interval > 0 and len(self.vis_video_lengths) > 0:
                self._setup_visualizer()  # 去初始化创建整个需要用的视频生成大模块组
            
        if self.one_logger is not None:
            self.one_logger.on_dataloader_init_end()   # 通告日志系统数据层全方面配置与实例化彻底完成的收尾口结束标志

        if self.one_logger is not None:
            self.one_logger.on_load_checkpoint_start()  # 现在去开启关于从过往断层接续学习加载数据的进度统计口
        if not self.is_lora_enabled:  # 当不是轻微调微处理工作的情况中即整个权重基础全部载入（普通基础框架下训练流程）
            # ================================= Standard (non-LoRA) model logic =================================  # 标准（无 LoRA）下的重构模型操作处理逻辑
            checkpoint_path = None  # 初始化一个空位置，指向应当继续去拉起的重构储存断头档
            
            if auto_resume and self.output_path:  # 开启了可以自己自动侦探搜索功能以及输出文件有预确定义
                # Auto resume: find latest checkpoint in logdir  # 从对应的存档文件夹搜索查询找到其中储存最新的那个迭代存盘处
                latest_checkpoint = self.find_latest_checkpoint(self.output_path)
                if latest_checkpoint:  # 如果没有丢失并且找到了断点头文件信息后
                    checkpoint_path = latest_checkpoint  # 指明接下来读取路线目标
                    if self.is_main_process:
                        print(f"Auto resume: Found latest checkpoint at {checkpoint_path}")  # 显示告诉用户找到继续操作节点头
                else:
                    if self.is_main_process:
                        print("Auto resume: No checkpoint found in logdir, starting from scratch")  # 即使搜遍并无记录那这就必定为第一次开启全新的生成，需要通知白板开始
            elif auto_resume:  # 搜不出 log 只有 auto resume 时没设保存处那就成了毫无根据和依托
                if self.is_main_process:
                    print("Auto resume enabled but no logdir specified, starting from scratch")  # 提议还是采用从开天地起头
            else:
                if self.is_main_process:
                    print("Auto resume disabled, starting from scratch")  # 因为命令拒绝干涉恢复动作那就也从起点造林启发
            
            if checkpoint_path is None:  # 当经历查探后这槽内仍然没有填补断点存储点
                if getattr(config, "generator_ckpt", False):  # 会去看看参数有没主观指定了一个用于导入覆盖的初始存参位置
                    # Explicit checkpoint path provided  # 使用确定化固定存向指定路径加载进行微调等
                    checkpoint_path = config.generator_ckpt
                    if self.is_main_process:
                        print(f"Using explicit checkpoint: {checkpoint_path}")

            if checkpoint_path:  # 如果确认要根据特定位置作解析引入工作
                if self.is_main_process:
                    print(f"Loading checkpoint from {checkpoint_path}")  # 报告已确定的要加载的基础模型断点
                checkpoint = torch.load(checkpoint_path, map_location="cpu")  # 将整体模型包载入内存
                
                # Load generator  # 加载核心生成器模型参数
                if "generator" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {checkpoint_path}")
                    self.model.generator.load_state_dict(checkpoint["generator"], strict=True)  # 要求严格匹配无遗漏键
                elif "model" in checkpoint:  # 如果是以别名存的
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {checkpoint_path}")
                    self.model.generator.load_state_dict(checkpoint["model"], strict=True)
                else:  # 若没有
                    if self.is_main_process:
                        print("Warning: Generator checkpoint not found.")  # 发出找不到生成器的警告
                
                # Load critic  # 尝试复现 Critic (判别器) 状态
                if "critic" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained critic from {checkpoint_path}")
                    self.model.fake_score.load_state_dict(checkpoint["critic"], strict=True)
                else:
                    if self.is_main_process:
                        print("Warning: Critic checkpoint not found.")
                
                # Load EMA  # 尝试同时复现滑动平均 EMA 模型组态
                if "generator_ema" in checkpoint and self.generator_ema is not None:
                    if self.is_main_process:
                        print(f"Loading pretrained EMA from {checkpoint_path}")
                    self.generator_ema.load_state_dict(checkpoint["generator_ema"])  # 载入之前存盘的 EMA 记录
                else:
                    if self.is_main_process:
                        print("Warning: EMA checkpoint not found or EMA not initialized.")  # 若未存在或不需要则跳过
                
                # For auto resume, always resume full training state  # 为真正的续约继续学习恢复全部原貌环境
                # Load optimizers  # 获取和重置以前对应优化器的历史动量、衰减状态
                if "generator_optimizer" in checkpoint:
                    if self.is_main_process:
                        print("Resuming generator optimizer...")
                    gen_osd = FSDP.optim_state_dict_to_load(  # 特别注意在 FSDP 引擎下需要调用内部特种的方法完成分布式权重的拼装对齐
                        self.model.generator,              # FSDP root module  # FSDP 模型根实体
                        self.generator_optimizer,          # newly created optimizer  # 新生造出来的空内容优化器实例
                        checkpoint["generator_optimizer"]  # optimizer state dict at save time  # 历史提取出来未切片原始版优化参数
                    )
                    self.generator_optimizer.load_state_dict(gen_osd)  # 注入还原好的适配当前环境结构的动量记录
                else:
                    if self.is_main_process:
                        print("Warning: Generator optimizer checkpoint not found.")
                
                if "critic_optimizer" in checkpoint:  # 对判别器采用相同的恢复对齐手续
                    if self.is_main_process:
                        print("Resuming critic optimizer...")
                    crit_osd = FSDP.optim_state_dict_to_load(
                        self.model.fake_score,
                        self.critic_optimizer,
                        checkpoint["critic_optimizer"]
                    )
                    self.critic_optimizer.load_state_dict(crit_osd)
                else:
                    if self.is_main_process:
                        print("Warning: Critic optimizer checkpoint not found.")
                
                # Load training step  # 获取当初保存的迭代次步标定从哪里再次延续
                if "step" in checkpoint:
                    self.step = checkpoint["step"]
                    if self.is_main_process:
                        print(f"Resuming from step {self.step}")
                else:
                    if self.is_main_process:
                        print("Warning: Step not found in checkpoint, starting from step 0.")

        if self.one_logger is not None:
            self.one_logger.on_load_checkpoint_end()  # 记录完并上报该关键节点步骤的结束（耗时）
        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference  # 在最早期几步先放弃计算 EMA 更新可以避免算力以及推断阶段无意义的重组开销
        # Note: This should be done after potential resume to avoid accidentally deleting resumed EMA  # 如果是刚续上来且没到门限则不应该意外洗掉刚艰难拼装的老 EMA
        if self.step < config.ema_start_step:
            self.generator_ema = None  # 在达到开始记录门限（例如 step=2000）之前，置空关闭

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)  # 设置主梯度裁剪阈值为防 Generator 梯度爆炸
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)  # 同样对 Critic
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)  # 获取用于扩大等效 batch 的物理梯度累加缓存步
        self.previous_time = None  # 记录上一步的秒表用来后续计步速
        
        # streaming training configuration  # 解析加载关于长序列流式并行串联生成训练配置设定
        self.streaming_training = getattr(config, "streaming_training", False)
        self.streaming_chunk_size = getattr(config, "streaming_chunk_size", 21)  # 设置每个生成片段内包含着几张图像(如这里固定为 21 帧一个 block)
        self.streaming_max_length = getattr(config, "streaming_max_length", 63)  # 流式能接受串联达到的最大总长度容忍 (如这里为 3 blocks * 21)
        
        # Create streaming training model if enabled  # 真正实例出实现自回归接驳的复杂模块
        if self.streaming_training:
            self.streaming_model = StreamingTrainingModel(self.model, config)  # 实例化基于基本算法底座构筑的高层流式串结 Wrapper
            if self.is_main_process:
                print(f"streaming training enabled: chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}")
        else:
            self.streaming_model = None  # 若不使用，则代表目前是普通训练模式
        
        # streaming training state (simplified)  # 初始化流式的简单状态变量机
        self.streaming_active = False  # Whether we're currently in a sequence  # 用来标志现在所处环境：是否正接在一部长期序列之中
        
        if self.is_main_process:  # 统揽全局后做最后的启动前各项参数指标自报展示回馈：
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            if self.gradient_accumulation_steps > 1:
                print(f"Effective batch size: {config.batch_size * self.gradient_accumulation_steps * self.world_size}")  # 报告真正发生质变的 Batch
            if self.streaming_training:
                print(f"streaming training enabled: chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}")
            if LOG_GPU_MEMORY:
                log_gpu_memory("After initialization", device=self.device, rank=dist.get_rank())  # 检测显卡驻留显存基线，防止后续训练中 OOM 无从查起内存泄漏
        
        if self.one_logger is not None:
            self.one_logger.on_train_start(train_iterations_start = self.step, train_samples_start = self.step * self.config.batch_size)  # 宣告日志上报开始正式迈入训练周期
        
    def _move_optimizer_to_device(self, optimizer, device):  # 定义个内部函数帮助快速倒换优化器动量张量所在设备环境
        """Move optimizer state to the specified device."""  # 为防止优化器动静不匹配产生的各种不可思议的 device mismatch error
        for state in optimizer.state.values():  # 取每一层的跟踪信息
            for k, v in state.items():  # 层内部具体的数据内容（诸如 exp_avg 等张量指标）
                if isinstance(v, torch.Tensor):  # 所有张量实体
                    state[k] = v.to(device)  # 做一次重绑定置入特定 device (如 当前工作的 GPU)
                    
    def find_latest_checkpoint(self, logdir):  # 定义辅助函数用于发掘给定的日志目录下存在的最晚（即内容最新）存放文件夹的名字
        """Find the latest checkpoint in the logdir."""
        if not os.path.exists(logdir):  # 如果根本这个上层建筑都不存在，那自然也没有内容
            return None
        
        checkpoint_dirs = []  # 建空阵保存有效候选项
        for item in os.listdir(logdir):  # 游历每一个在其中的元素
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):  # 如果是符合规则并且是个明确的资料文件夹的才能考察
                try:
                    # Extract step number from directory name  # 将前缀剥离直接解析该储存所代表是第几次迭代留存
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)  # 转整型供比较
                    checkpoint_path = os.path.join(logdir, item, "model.pt")  # 探查该目录下是否存放主模型存盘内容
                    if os.path.exists(checkpoint_path):
                        checkpoint_dirs.append((step, checkpoint_path))  # 如文件都真实存在那这个算成可用候选放入
                except ValueError:
                    continue  # 当解析出的不合规数值时报错被捕获，然后放过这一个误闯对象
        
        if not checkpoint_dirs:  # 当淘金结束后依然空手，那么返回虚空
            return None
        
        # Sort by step number and return the latest one  # 当有了一堆历史候选就可以将它们按照记录排序取得当前能用最好的状态
        checkpoint_dirs.sort(key=lambda x: x[0])  # 依据步数从小升序排列
        latest_step, latest_path = checkpoint_dirs[-1]  # 抽取位于阵列队伍最后面（最末尾，既时间最新的存盘路径）
        return latest_path  # 给出去这个最终最可信的最近目标位置

    def get_all_checkpoints(self, logdir):  # 定义个相似辅佐用来收集整个目录里面能被认定是合法的全体内容群（便于管理空间利用如只留最近几个以防撑爆硬盘）
        """Get all checkpoints in the logdir sorted by step number."""
        if not os.path.exists(logdir):
            return []
        
        checkpoint_dirs = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    # Extract step number from directory name
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)
                    checkpoint_dir_path = os.path.join(logdir, item)
                    checkpoint_file_path = os.path.join(checkpoint_dir_path, "model.pt")
                    if os.path.exists(checkpoint_file_path):
                        checkpoint_dirs.append((step, checkpoint_dir_path, item))  # 这次多记录一个方便删除等整目录操作（存下外层文件夹绝对路径及纯名）
                except ValueError:
                    continue
        
        # Sort by step number (ascending order)
        checkpoint_dirs.sort(key=lambda x: x[0])
        return checkpoint_dirs  # 给出完整有序列

    def cleanup_old_checkpoints(self, logdir, max_checkpoints):  # 自动巡护处理超额储存以免硬盘爆满引起直接 crash 以及妨碍其后运行存储
        """Remove old checkpoints if the number exceeds max_checkpoints.
        Only the main process performs the actual deletion to avoid race conditions
        in distributed training.
        """  # 限制该清除高危操只能在主进位发生以绝对禁杜发生多进程互相竞争删空文件的崩溃事故导致数据损坏无法还原的可能
        if max_checkpoints <= 0:  # 如果配置设定表示不需要任何自动清理限度那么直接忽略离开
            return
        
        # Only main process should perform cleanup to avoid race conditions  # 第二次锁保证，只允许 0 号去删其他都不许过界
        if not self.is_main_process:
            return
            
        checkpoints = self.get_all_checkpoints(logdir)  # 通过前人函数列全所有留存
        if len(checkpoints) > max_checkpoints:  # 若发现真的超出红线阈值门限
            # Calculate how many to remove  # 算出应当除去几位老住客才能腾出满足符合要求的配置安全总数空间
            num_to_remove = len(checkpoints) - max_checkpoints
            checkpoints_to_remove = checkpoints[:num_to_remove]  # Remove oldest ones  # 利用排序的特性，从队伍最开始就是最早（步数低老存盘）将其选取开刀
            
            print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints, removing {num_to_remove} oldest ones (keeping {max_checkpoints})")
            
            import shutil  # 调用操作系统的模块以便做深层彻底文件夹清除
            removed_count = 0  # 追踪记录清理完成数
            for step, checkpoint_dir_path, dir_name in checkpoints_to_remove:  # 循环每个需要铲除的对象
                try:
                    print(f"  Removing: {dir_name} (step {step})")
                    shutil.rmtree(checkpoint_dir_path)  # 递归无留情面将所指向之整个含存盘和额外配置文件数据尽数毁灭抹去
                    removed_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to remove checkpoint {dir_name}: {e}")  # 偶尔磁盘使用可能遭到锁会清理失败产生无碍运行的不快只出日志
            
            print(f"Checkpoint cleanup completed: removed {removed_count}/{num_to_remove} old checkpoints")  # 回报总完成数
        else:
            if len(checkpoints) > 0:
                print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints (max: {max_checkpoints}, no cleanup needed)")  # 并未出危报不动作的宣告

    def _get_switch_frame_index(self, max_length=None):  # 设置或决定一个关于条件/话题切转点的方法（当一个长句子要切入另一段描述）决定哪一帧发生转变 
        if getattr(self.config, "switch_mode", "fixed") == "random":  # 可以选择是配置里的设定让转变帧充满不固定性（更强的学习健壮度分布）
            block = self.config.num_frame_per_block
            min_idx = self.config.min_switch_frame_index
            max_idx = self.config.max_switch_frame_index
            if min_idx == max_idx:
                switch_idx = min_idx  # 既然最大最小门限一致，就直接是它自己
            else:
                choices = list(range(min_idx, max_idx, block))  # 利用分块数，确保每一次大扭转只在块缝拼接那一段开头进行以免单一块截两段不好预测
                if max_length is not None:
                    choices = [choice for choice in choices if choice < max_length]  # 要保证这个转变点必须在此视频最远帧的以内！
                
                if len(choices) == 0:  # 但是剔完了发现没的选，或者都不符
                    if max_length is not None:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")  # 这个肯定代表了严重的错误，要让用户晓得
                    else:
                        switch_idx = block  # 提供缺省 fallback 容灾策略
                else:
                    if dist.get_rank() == 0:
                        switch_idx = random.choice(choices)  # 这个决策需要让全部环境知道而且必须绝对一致，只给主节点权限抽取结果签子并向群里通报
                    else:
                        switch_idx = 0  # placeholder; will be overwritten by broadcast  # 其他节点的留着给占座空着（反正是要被覆盖的）
                switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
                dist.broadcast(switch_idx_tensor, src=0)  # 使用群体大本营向下所有附从派发
                switch_idx = switch_idx_tensor.item()
        elif getattr(self.config, "switch_mode", "fixed") == "fixed":  # 如果是不带变化的硬写入式死方法
            switch_idx = getattr(self.config, "fixed_switch_index", 21)
            if max_length is not None:
                assert max_length > switch_idx, f"max_length {max_length} is not greater than switch_idx {switch_idx}"
        elif getattr(self.config, "switch_mode", "fixed") == "random_choice":  # 如果是有序的可选项中择出来的方法（区别全完全放任的范围随机）
            switch_choices = getattr(self.config, "switch_choices", [])
            if len(switch_choices) == 0:
                raise ValueError("switch_choices is empty")
            else:
                if max_length is not None:
                    switch_choices = [choice for choice in switch_choices if choice < max_length]
                    if len(switch_choices) == 0:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")
                
                if dist.get_rank() == 0:
                    switch_idx = random.choice(switch_choices)
                else:
                    switch_idx = 0
            switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
            dist.broadcast(switch_idx_tensor, src=0)
            switch_idx = switch_idx_tensor.item()
        else:
            raise ValueError(f"Invalid switch_mode: {getattr(self.config, 'switch_mode', 'fixed')}")
        return switch_idx


    def save(self):  # 定义保存模型状态的主函数
        print("Start gathering distributed model states...")
        if getattr(self, 'one_logger', None) is not None and self.is_main_process:
            self.one_logger.on_save_checkpoint_start(global_step=self.step)  # 通知上报系统开始存档过程

        if self.is_lora_enabled:  # 若是使用了高效参数微调 LoRA 模式
            gen_lora_sd = self._gather_lora_state_dict(  # 从包裹满分布式的封装内单独捞出 LoRA 层的权重字典
                self.model.generator.model)
            crit_lora_sd = self._gather_lora_state_dict(
                self.model.fake_score.model)

            state_dict = {  # 在纯 LoRA 存盘时
                "generator_lora": gen_lora_sd,  # 仅存少量附加权重网络即可，巨大底模无需重写
                "critic_lora": crit_lora_sd,
                "step": self.step,  # 并保存步数
            }
        else:  # 全量非 LoRA 保存时
            with FSDP.state_dict_type(  # 启动 FSDP 字典提取上下文环境以便将切碎的数据复原为完整大块
                self.model.generator,
                StateDictType.FULL_STATE_DICT,  # 要求抽取拼凑一整个完整模型结构，而不是单独节点的私有分片
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),  # 规定拼凑汇总的只由主节点执行且将显存转存入内存避免爆 OOM 炸满
                FullOptimStateDictConfig(rank0_only=True),          # newly added  # 对优化器信息作同样汇总限制操作
            ):
                generator_state_dict  = self.model.generator.state_dict()  # 获取组装后的全量权重
                generator_opim_state_dict = FSDP.optim_state_dict(self.model.generator,  # 取出相应的优化器动量信息
                                                self.generator_optimizer)

            with FSDP.state_dict_type(  # 对于批评者（Discriminator/Critic）同样执行相同的组合提取过程
                self.model.fake_score,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True),          # newly added
            ):
                critic_state_dict  = self.model.fake_score.state_dict()  
                critic_opim_state_dict = FSDP.optim_state_dict(self.model.fake_score,
                                                self.critic_optimizer)

            if self.config.ema_start_step < self.step and self.generator_ema is not None:  # 如果保存点在 EMA 启动之后
                state_dict = {
                    "generator": generator_state_dict,
                    "critic": critic_state_dict,
                    "generator_ema": self.generator_ema.state_dict(),  # 则连带将累加 EMA 包字典存盘备用查验/直接部署推演
                    "generator_optimizer": generator_opim_state_dict,
                    "critic_optimizer": critic_opim_state_dict,
                    "step": self.step,
                }
            else:  # 没有 EMA 过的情况
                state_dict = {
                    "generator": generator_state_dict,
                    "critic": critic_state_dict,
                    "generator_optimizer": generator_opim_state_dict,
                    "critic_optimizer": critic_opim_state_dict,
                    "step": self.step,
                }

        if self.is_main_process:  # 将组装就位好的存盘结构字典真的物理写入挂起的存储
            checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")  # 根据最新的一步计数定名子级文件夹
            os.makedirs(checkpoint_dir, exist_ok=True)  # 若无则建目录
            checkpoint_file = os.path.join(checkpoint_dir, "model.pt")  # 具体的数据结构保存位
            torch.save(state_dict, checkpoint_file)  # 将结构内容正式落盘序列化至文件系统
            print("Model saved to", checkpoint_file)
            
            # Cleanup old checkpoints if max_checkpoints is set  # 清理老旧盘操作
            max_checkpoints = getattr(self.config, "max_checkpoints", 0)  # 获取盘数量阈值
            if max_checkpoints > 0:
                self.cleanup_old_checkpoints(self.output_path, max_checkpoints)  # 执行多退少补策略删前缀最老记录

        torch.cuda.empty_cache()  # 无论主次节点，大家都把刚刚为做重组抽调的碎渣和临时工丢掉缓解压力
        import gc
        gc.collect()  # 触发 Python 原生的 GC 让不用或者不指向引用的垃圾强制扔掉清理
    
        if self.one_logger is not None:
            self.one_logger.on_save_checkpoint_success(global_step=self.step)  # 发文宣称这笔保存完毕
            self.one_logger.on_save_checkpoint_end(global_step=self.step)  # 终止打卡计时器段

    def fwdbwd_one_step(self, batch, train_generator):  # 最核心关键之一：定义走一次 Forward 并带着反向去收 Gradient (梯度的) 方法结构
        self.model.eval()  # prevent any randomness (e.g. dropout)  # 因为计算中需要去对标一些行为且依赖 DMD 方法特殊的伪反向生成，不可以用训练态

        if self.step % 5 == 0:  # 周期清理垃圾显存碎片防 OOM
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts  # 第一阶段: 从送来的材料中解出描述提示文本 (Prompt)
        text_prompts = batch["prompts"]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)  # 从设配的清单上得知产出的视频/图之宏观框寸尺度 
        image_or_video_shape[0] = batch_size  # 根据数据装载器反馈动态决定实际该这把造多宽（替换设定的 Batch Size 位置值）

        # Step 2: Extract the conditional infos  # 第二阶段: 获取指导内容控制潜信号(cond embedding) 等相关附加物
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(  # 用挂载的特征映射文本器（例如 T5 或 CLIP Text）对文本字串提出表象表示 (Representation)
                text_prompts=text_prompts)

            if self.config.i2v and "img" in batch:  # 对专职 I2V 且确实带初始约束画面的请求做出特化对位
                img = batch["img"].to(self.device)  # 取画并丢进 GPU
                conditional_dict["clip_fea"] = self.model.image_encoder(img)  # 使用图表层抽取对齐表意 CLIP 特征嵌入
                conditional_dict["y"] = self.model.vae.run_vae_encoder(img)  # 取潜通道原始压码图像像素块 (由 VAE 操手)

            if not getattr(self, "unconditional_dict", None):  # 防止运算挥霍性能设好如果已经有备好的零号底提示（负面/无引导态）
                unconditional_dict = self.model.text_encoder(  # 无底的话当场制作一组和批次对等地空或反选无条件引导字典提供给无约束推断 CFG 的偏离使用
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}  # 切除任何无用反推图图计算图以免爆存
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict  # 收入自建静态缓冲下次白嫖
            else:
                unconditional_dict = self.unconditional_dict  # 下次取便有直接套用了

        # Step 3: Store gradients for the generator (if training the generator)  # 第三阶段: 生成并捕捉模型要修改调整的落差度
        if train_generator:  # 通过布尔标志来判断此时我们这一步应该打分且修正是给予主角生成者还是打分员判别者
            generator_loss, generator_log_dict = self.model.generator_loss(  # 去调用外挂在自己身上由 DMD 特指定义的生成生手模型 Loss 求法逻辑计算落差
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=None,  # 不用干净原视频因为这是一种从无到有不需 GT 比对训练（即蒸馏特长所在无需原带真视频跑这趟）
                initial_latent=None
            )

            # Scale loss for gradient accumulation and backward  # 在算子向回退前先摊平这个值，这是面对多步堆叠小 Batch 的正确求导操作
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            scaled_generator_loss.backward()  # 使 PyTorch 发动全网微分传导反压，梯次灌入那些勾有需要修习的权位
            if LOG_GPU_MEMORY:
                log_gpu_memory("After train_generator backward pass", device=self.device, rank=dist.get_rank())
            # Return original loss for logging  # 为回显做日志信息补登记录原装损失
            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation  # 这些后续真正 Step 的时候再去抓取补上

            return generator_log_dict  # 立刻抽离
        else:
            generator_log_dict = {}

        critic_loss, critic_log_dict = self.model.critic_loss(  # 进入不调参只考察培训 Critic（假评判裁判的鉴定水平业务，也是 DMD 体系灵魂：双网左手打右手）
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=None,
            initial_latent=None
        )

        # Scale loss for gradient accumulation and backward  # 为裁判也算出来等效损失值以便叠加求均
        scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
        scaled_critic_loss.backward()  # 使损失量穿透网络产生纠正调整方向力值
        if LOG_GPU_MEMORY:
            log_gpu_memory("After train_critic backward pass", device=self.device, rank=dist.get_rank())
        # Return original loss for logging
        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation

        return critic_log_dict  # 携带结果汇报

    def generate_video(self, pipeline, num_frames, prompts, image=None):  # 定义纯应用性的对外产生结果辅助包装用函数
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent  # 如果带了图来就把它当作源生视频的第一落笔点存作初始潜信源
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)  # 根据 Batch 的需要扩张排版铺成一组列
            sampled_noise = torch.randn(  # 给这之后要产生的数张（原时长数 - 1 即为需补充长度）留位建高斯噪音作为推演的素材坯子
                [batch_size, num_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:  # 当没有图像要求全是凭空用词构图时
            initial_latent = None
            sampled_noise = torch.randn(  # 那就直接申请全长度等长的彻底纯色噪音素材板
                [batch_size, num_frames, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )
        with torch.no_grad():
            video, _ = pipeline.inference(  # 去调用之前定义的整套 pipeline 工艺线通过推演来解噪音最终出真形
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
            )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0  # 变换张量把各对应维度重整回常规视频三基色与帧位布局制式的正统展现数据矩阵类型
        pipeline.vae.model.clear_cache()  # 防止因产生巨物产生的 VAE 缓冲区未释放
        return current_video  # 带回可直接操作输出的结果
    

    def generate_video_with_switch(self, pipeline, num_frames, prompts, switch_prompts, switch_frame_index, image=None):  # 对带突兀句意跳出转折点生成的专用功能器方法支持
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, num_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, num_frames, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )
        with torch.no_grad():
            video, _ = pipeline.inference(  # 特别注意此下它调用的传参里面比上面多挂递了一个目标短词及其预设会话转变交接卡点的那个帧步数字参量
                noise=sampled_noise,
                text_prompts_first=prompts,
                text_prompts_second=switch_prompts,
                switch_frame_index=switch_frame_index,
                return_latents=True
            )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        pipeline.vae.model.clear_cache()
        return current_video

    def start_new_sequence(self):  # 该段是开启长序列生成的一小包动作汇总，流式必备
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] start_new_sequence called")
        
        if LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: Before start_new_sequence", device=self.device, rank=dist.get_rank())
        
        # Fetch a new batch  # 重取一段全新素材拉开始的帷幕头
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] start_new_sequence: fetch new batch")
        batch = next(self.dataloader)  # 从游走列表池借走一条记录

        # Prepare conditional information  # 消化与整顿控制用源信息以成建制潜序列
        text_prompts = batch["prompts"]
        if self.config.i2v:
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(  # 从提供的带图资源里抽出最后（由于一般只有第一张其实等同于那个图像）存作为潜值起始基点推入硬件阵
                device=self.device, dtype=self.dtype)
        else:
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] Setting up sequence: batch_size={batch_size}, i2v={self.config.i2v}")
            print(f"[SeqTrain-Trainer] image_or_video_shape={image_or_video_shape}")
        
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)  # 为当前新取文字转释对应条件字典结构件给将要去往推移训练循环中服务
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Created and cached conditional_dict")
            if not getattr(self, "unconditional_dict", None):  # 同理，准备一份空信息的对照用无条件字典，只用造一次存下就不反复制了
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] Created and cached unconditional_dict")
            else:
                unconditional_dict = self.unconditional_dict
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: After text encoding", device=self.device, rank=dist.get_rank())
        
        if self.streaming_model.possible_max_length is not None:  # 如果配置里规定了一组随机长度列表，那每次开启新剧集长度都是一次抽签决定的
            # Ensure all processes choose the same length  # 为了在多网环境中切分并保持同步，长度摇号必须群组内保持严格一致
            if dist.is_initialized():
                if dist.get_rank() == 0:  # 让主进程独立完成掷骰子
                    import random
                    selected_idx = random.randint(0, len(self.streaming_model.possible_max_length) - 1)
                else:
                    selected_idx = 0  # 其他人等消息
                selected_idx_tensor = torch.tensor(selected_idx, device=self.device, dtype=torch.int32)
                dist.broadcast(selected_idx_tensor, src=0)  # 发牌
                selected_idx = selected_idx_tensor.item()
            else:
                import random
                selected_idx = random.randint(0, len(self.streaming_model.possible_max_length) - 1)  # 单机情况下自己抛骰子
            
            temp_max_length = self.streaming_model.possible_max_length[selected_idx]  # 从备选数组抓出本次敲定的时序长
        else:
            temp_max_length = self.streaming_model.max_length  # 无随机池子则直接使用绝对固定最大长
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Model] Selected temporary max length: {temp_max_length} (from {self.streaming_model.possible_max_length})")
        

        # Handle DMD Switch related information  # 处理有关句子或者描述场景发生反转/切换的相关逻辑配套物料
        switch_conditional_dict = None
        switch_frame_index = None
        if isinstance(self.model, DMDSwitch) and "switch_prompts" in batch:  # 只在使用专门包含此机制网络且数据也附赠了跳脱提示才启动
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Processing DMDSwitch info")
                
            with torch.no_grad():
                switch_conditional_dict = self.model.text_encoder(  # 预编码作为目标导向的新一句转折点后所用文本指引向量
                    text_prompts=batch["switch_prompts"]
                )
            switch_frame_index = self._get_switch_frame_index(temp_max_length)  # 调用前文提到的取得跳转帧索位的方法
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] switch_frame_index={switch_frame_index}")
            
            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                log_gpu_memory(f"streaming Training: After switch text encoding", device=self.device, rank=dist.get_rank())
        
        # Set up the sequence  # 将处理并集结好一切需要的配置装配注入启动长片段工作列车班次调度
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] Calling streaming_model.setup_sequence")
            
        self.streaming_model.setup_sequence(  # 交由下属类实操部署准备
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            initial_latent=image_latent,
            switch_conditional_dict=switch_conditional_dict,
            switch_frame_index=switch_frame_index,
            temp_max_length=temp_max_length,
        )
        
        self.streaming_active = True  # 正式宣告该车次已组装完成正在运行生产中标识开启
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] streaming training sequence setup completed")
            
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: After sequence setup", device=self.device, rank=dist.get_rank())

    def fwdbwd_one_step_streaming(self, train_generator):  # 针对开启流式产生的大长流切分成单个流传式步进推进处理和梯度回收方法
        """Forward/backward pass using the new StreamingTrainingModel for serialized training"""
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 5 == 0:
            torch.cuda.empty_cache()

        # If no active sequence, start a new one  # 此状态机若是待命停机那必须在要求运转前呼叫开始填列开机
        if not self.streaming_active:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] No active sequence, starting new one")
            self.start_new_sequence()  # 即执行上面那个取素材准备发车方法
        
        # Check whether we can generate more chunks  # 在干活前先检视下这个接力序列是不是已经跑到尽头顶天无空间可切片了
        if not self.streaming_model.can_generate_more():
            # Current sequence is finished; start a new one  # 到此头即完，直接弃废老车皮再从数据集拔新数据启新局
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Current sequence completed, starting new one")
            self.streaming_active = False
            self.start_new_sequence()
        
        self.kv_cache_before_generator_rollout = None  # 防止在步进时各种 KV Cache 的前代残留指针，全部显式断清干干净净防泄漏
        self.kv_cache_after_generator_rollout = None
        self.kv_cache_after_generator_backward = None
        self.kv_cache_before_critic_rollout = None
        self.kv_cache_after_critic_rollout = None
        self.kv_cache_after_critic_backward = None
        
        if train_generator:  # 进入对 Generator 提供的一系列产生、算亏评估、返回误差的操作环
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Training generator: generating next chunk")

            train_first_chunk = getattr(self.config, "train_first_chunk", False)  # 查阅参数本确定不练头一片的内容？
            if train_first_chunk:  # 如果包含有从第一手就强开计算的权限
                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=True)  # 要求产生一块可以提供溯源微积分倒算的片段数据
            else:  # 平常见最多的是不对开头那帧起求（可能那是真图像本身）
                current_seq_length = self.streaming_model.state.get("current_length")  # 获取现阶段已经走了多长距离
                if current_seq_length == 0:  # 也就是说当前确实站在刚刚出发第一步！
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}, generate first chunk")
                    generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)  # 这第一步强制闭关梯求无状态生成直接推进缓存积累上下文

                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=True)  # 这才去产生那些随后承接能够溯回归纳的普通长节带梯度节点片
            
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}")

            # Compute generator loss  # 把取得的可追溯片段塞入评价系统算出和目标的差距数值
            generator_loss, generator_log_dict = self.streaming_model.compute_generator_loss(
                chunk=generated_chunk,
                chunk_info=chunk_info
            )

            # Scale loss for gradient accumulation and backward  # 为多次集叠平权防止数值暴增
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[DEBUG] Scaled generator loss: {scaled_generator_loss.item()}")

            try:
                scaled_generator_loss.backward()  # 向着所有参数推发指导力度信息
            except RuntimeError as e:
                raise  # 出事就爆不硬接

            generator_log_dict.update({
                "generator_loss": generator_loss,
                "generator_grad_norm": torch.tensor(0.0, device=self.device),  # 占位符留待积累打满后正式写落
            })
            
            return generator_log_dict
        else:  # 若此次并非考核生成能力而是测绘及训练那个负责把关审判官（网络对鉴模块）
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Training critic: generating next chunk")

            train_first_chunk = getattr(self.config, "train_first_chunk", False)  # 同样先查开头规矩
            if train_first_chunk:
                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)  # 不同点在于判别器没须要用梯度反馈到生成时即直接掐死反推节省计算开销并避免串网越界干扰
            else:
                current_seq_length = self.streaming_model.state.get("current_length")
                if current_seq_length == 0:
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}, generate first chunk")
                    generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)  # 一样无梯无感路过

                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)  # 给 Critic 作靶用的视频也一律无感生成，只需产出结果物供其评价比对
            
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}")

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Generated chunk shape: {generated_chunk.shape}")
                print(f"[SeqTrain-Trainer] Generated chunk requires_grad: {generated_chunk.requires_grad}")
            
            if generated_chunk.requires_grad:  # 为保险再硬摘除一遍求导图确保安全拆离
                generated_chunk = generated_chunk.detach()

            # Compute critic loss  # 放进批评考核流程对其预测判别水准做评分（越近真实的鉴赏力度，它得到Loss越小表示它越称职做火眼金睛）
            critic_loss, critic_log_dict = self.streaming_model.compute_critic_loss(
                chunk=generated_chunk,
                chunk_info=chunk_info
            )
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Critic loss: {critic_loss.item()}")
            
            # Scale loss for gradient accumulation and backward  # 下发平推该指导
            scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
            scaled_critic_loss.backward()  # 使评价员长了见识升了阶
            
            critic_log_dict.update({
                "critic_loss": critic_loss,
                "critic_grad_norm": torch.tensor(0.0, device=self.device),
            })
            
            return critic_log_dict

    def train(self):  # 重头戏！这就是系统调遣所有的底层动作进行循环主轴！即 Train Epoch 循环全代码起始
        start_step = self.step  # 将当面起步计号拿稳
        try:
            while True:  # 开启不见南墙不落泪的长跑引擎直到手动 Break
                # Check if we should train generator on this optimization step  # 确认这场交锋是否轮到了被考核者的 Generator (例如某些 GAN 论文写让对抗训练判别方多走几步才放生成方走)
                TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0  # 通过比例计算是否准其一练
                if LOG_GPU_MEMORY:
                    log_gpu_memory(f"Before training", device=self.device, rank=dist.get_rank())
                
                if dist.get_rank() == 0 and DEBUG:
                    print(f"[Debug] Step {self.step}: switch_mode={getattr(self.config,'switch_mode','fixed')}")

                if self.one_logger is not None:
                    self.one_logger.on_train_batch_start()  # 开始计时录影打点报单轮起步

                if self.streaming_training:  # 如果切入了属于超长段的 Streaming 方式，会有特定的聚合累积步骤流
                    # Zero-out all optimizer gradients  # 首要任务，将残留过往打磨痕迹旧梯度清白纸
                    if TRAIN_GENERATOR:
                        self.generator_optimizer.zero_grad(set_to_none=True)  # 直接以释放张量 None 取代塞零更节电省内存
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    
                    # Whole-cycle gradient accumulation loop  # 进入完整闭环内的积累期轮循
                    accumulated_generator_logs = []
                    accumulated_critic_logs = []
                    
                    for accumulation_step in range(self.gradient_accumulation_steps):  # 按所规定等效变扩 Batch 幅度执行走几次存几次
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Whole-cycle accumulation step {accumulation_step + 1}/{self.gradient_accumulation_steps}")
                        
                        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and accumulation_step == 0:
                            log_gpu_memory(f"streaming Training Step {self.step}: Before whole-cycle forward/backward", device=self.device, rank=dist.get_rank())
                        
                        # Train generator (if needed)  # 操作 Generator 的流步推进回压获取亏量并收账记录
                        if TRAIN_GENERATOR:
                            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                                print(f"[SeqTrain-Trainer] Accumulation step {accumulation_step + 1}: Training generator")
                            extra_gen = self.fwdbwd_one_step_streaming(True)  # 调用特定的流产生法带参 True 表明生成系执行收回记录表落脚留痕
                            accumulated_generator_logs.append(extra_gen)
                        
                        # Train critic  # 同理判别也是要来一遍测其眼界
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Accumulation step {accumulation_step + 1}: Training critic")
                        extra_crit = self.fwdbwd_one_step_streaming(False)  # 带 False 表示考核判别业务段落获取差单保存
                        accumulated_critic_logs.append(extra_crit)
                        
                        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and accumulation_step == 0:
                            log_gpu_memory(f"streaming Training Step {self.step}: After whole-cycle forward/backward", device=self.device, rank=dist.get_rank())
                    
                    # Compute grad norm and update parameters  # 此时累加器装满，算梯度限流剪短毛刺以防偏坡并执行实质修改推进模型进化步子
                    if TRAIN_GENERATOR:
                        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)  # 削平爆炸性陡峰的极限值维持平稳训练
                        generator_log_dict = merge_dict_list(accumulated_generator_logs)  # 把那一堆打分评价单揉合成最终统计报表
                        generator_log_dict["generator_grad_norm"] = generator_grad_norm  # 在刚被占位的孔洞记录下这次被消峰前真正的数值作为观察点
                        
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Generator training completed, grad_norm={generator_grad_norm.item()}")
                        
                        self.generator_optimizer.step()  # 实实在在往下走一步跨越更新模型参数
                        if self.generator_ema is not None:
                            self.generator_ema.update(self.model.generator)  # 让 EMA 也跟上来记下此刻的背影
                    else:
                        generator_log_dict = {}
                    
                    critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)  # 裁判也削峰保护一下
                    critic_log_dict = merge_dict_list(accumulated_critic_logs)
                    critic_log_dict["critic_grad_norm"] = critic_grad_norm
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] Critic training completed, grad_norm={critic_grad_norm.item()}")
                    
                    self.critic_optimizer.step()  # 裁决权也实走一步
                    
                    if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                        log_gpu_memory(f"streaming Training Step {self.step}: After optimizer steps", device=self.device, rank=dist.get_rank())
                    
                    # Increase step count
                    self.step += 1  # 这一总的流式循环交替博弈大步正式结束迈向下一环
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] streaming training step completed: step={self.step}")
                        if hasattr(self, 'streaming_model') and self.streaming_model is not None:
                            current_seq_length = self.streaming_model.state.get("current_length", 0)
                            print(f"[SeqTrain-Trainer] Current sequence length: {current_seq_length}/{self.streaming_model.max_length}")
                            
                    if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                        log_gpu_memory(f"streaming Training Step {self.step}: Training step completed", device=self.device, rank=dist.get_rank())
                else:  # 若是普通的普通一次性即崩型的前馈和回馈（短生成非长切流）
                    if TRAIN_GENERATOR:
                        self.generator_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    
                    # Whole-cycle gradient accumulation loop
                    accumulated_generator_logs = []
                    accumulated_critic_logs = []
                    
                    for accumulation_step in range(self.gradient_accumulation_steps):
                        batch = next(self.dataloader)  # 取一次资料直接一把梭
                        
                        # Train generator (if needed)
                        if TRAIN_GENERATOR:
                            extra_gen = self.fwdbwd_one_step(batch, True)  # 普通型方法
                            accumulated_generator_logs.append(extra_gen)
                        
                        # Train critic
                        extra_crit = self.fwdbwd_one_step(batch, False)
                        accumulated_critic_logs.append(extra_crit)
                    
                    # Compute grad norm and update parameters
                    if TRAIN_GENERATOR:
                        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                        generator_log_dict = merge_dict_list(accumulated_generator_logs)
                        generator_log_dict["generator_grad_norm"] = generator_grad_norm
                        
                        self.generator_optimizer.step()
                        if self.generator_ema is not None:
                            self.generator_ema.update(self.model.generator)
                    else:
                        generator_log_dict = {}
                    
                    critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                    critic_log_dict = merge_dict_list(accumulated_critic_logs)
                    critic_log_dict["critic_grad_norm"] = critic_grad_norm
                    
                    self.critic_optimizer.step()

                    # Increment the step since we finished gradient update
                    self.step += 1

                if self.one_logger is not None:
                    self.one_logger.on_train_batch_end()  # 循环结束报时打卡

                # Create EMA params (if not already created)  # 等跑过了 EMA 免算期限制，开启并生成容器接受更新了
                if (self.step >= self.config.ema_start_step) and \
                        (self.generator_ema is None) and (self.config.ema_weight > 0):
                    if not self.is_lora_enabled:  # 但 LoRA 不享受此待遇依然旁观
                        self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
                        if self.is_main_process:
                            print(f"EMA created at step {self.step} with weight {self.config.ema_weight}")
                    else:
                        if self.is_main_process:
                            print(f"EMA creation skipped at step {self.step} (disabled in LoRA mode)")

                # Save the model  # 定期存盘保护不翻车重跑
                if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                    torch.cuda.empty_cache()  # 抽掉水分
                    self.save()  # 主调刚刚的那些打包全存与删除过期文件的综合方法
                    torch.cuda.empty_cache()

                # Logging  # 数据板整理填涂上大屏反馈
                if self.is_main_process:
                    wandb_loss_dict = {}
                    if TRAIN_GENERATOR and generator_log_dict:
                        wandb_loss_dict.update(
                            {
                                "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                                "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                                "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()  # 以及对 DMD特有的惩罚梯度作显
                            }
                        )


                    wandb_loss_dict.update(
                        {
                            "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                            "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                        }
                    )
                    if not self.disable_wandb:  # 云端面板发送心跳上传各维度图表值域
                        wandb.log(wandb_loss_dict, step=self.step)

                if self.step % self.config.gc_interval == 0:  # 周期大清洁整理内存环境防泄溢（GC=GarbageCollection）
                    if dist.get_rank() == 0:
                        logging.info("DistGarbageCollector: Running GC.")
                    gc.collect()
                    torch.cuda.empty_cache()

                if self.is_main_process:  # 将本地耗时状况向本地命令行打印或云端推送
                    current_time = time.time()
                    iteration_time = 0 if self.previous_time is None else current_time - self.previous_time
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": iteration_time}, step=self.step)
                    self.previous_time = current_time
                    # Log training progress
                    if TRAIN_GENERATOR and generator_log_dict:
                        print(f"step {self.step}, per iteration time {iteration_time}, generator_loss {generator_log_dict['generator_loss'].mean().item()}, generator_grad_norm {generator_log_dict['generator_grad_norm'].mean().item()}, dmdtrain_gradient_norm {generator_log_dict['dmdtrain_gradient_norm'].mean().item()}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}")
                    else:
                        print(f"step {self.step}, per iteration time {iteration_time}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}")

                # ---------------------------------------- Visualization ---------------------------------------------------
                # 执行周期性推演看当前进展成果阶段
                if self.vis_interval > 0 and (self.step % self.vis_interval == 0):
                    if self.one_logger is not None:
                        self.one_logger.on_validation_start()

                    try:
                        self._visualize()  # 触发造小电影功能以便人眼查看评估现今网络掌握规律虚实
                    except Exception as e:
                        print(f"[Warning] Visualization failed at step {self.step}: {e}")  # 偶尔造失败不打断主集继续深造
                
                    if self.one_logger is not None:
                        self.one_logger.on_validation_end()
                
                if self.step > self.config.max_iters:  # 探知若走到定数末路尽头则终结跳离循环跑道结束整个旅程
                    break

            if self.one_logger is not None:
                self.one_logger.on_train_end()  # 大终场打卡下线
                self.one_logger.on_app_end()
        
        except Exception as e:  # 这个用于网捕所有的中途横祸，一旦非预估的崩溃出现抓其错误和所在
            if self.is_main_process:
                print(f"[ERROR] Training crashed at step {self.step} with exception: {e}")
                print(f"[ERROR] Exception traceback:", flush=True)  # 将爆错堆栈强制打印且不得缓充
                import traceback
                traceback.print_exc()
        finally:  # 代表无论正常结束还是炸飞中止都在退出时保底清线索
            # Clean up resources
            if self.one_logger is not None:
                try:
                    self.one_logger.on_train_end()
                    self.one_logger.on_app_end()
                except Exception as cleanup_e:
                    if self.is_main_process:
                        print(f"[WARNING] Failed to clean up one_logger: {cleanup_e}")


    def _configure_lora_for_model(self, transformer, model_name):  # （附属功用）：只给目标内需用的 Transformer 内特定网络构架贴上/套皮 LoRA 的微参
        """Configure LoRA for a WanDiffusionWrapper model"""
        # Find all Linear modules in WanAttentionBlock modules
        target_linear_modules = set()  # 等着收集的目标清单池子
        
        # Define the specific modules we want to apply LoRA to  # 限定一下范围要贴层去哪里贴
        if model_name == 'generator':
            adapter_target_modules = ['CausalWanAttentionBlock']  # 由于主模型带有因果结构块
        elif model_name == 'fake_score':
            adapter_target_modules = ['WanAttentionBlock']  # 而打分的鉴赏模型只是纯靠注意区块评判即可不需要前因后果追溯
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        for name, module in transformer.named_modules():  # 挖出当前骨干网络上的每个细零部件端看
            if module.__class__.__name__ in adapter_target_modules:  # 是否对上了指定的重点部件类型（上述）
                for full_submodule_name, submodule in module.named_modules(prefix=name):  # 这还没完，大部件底下找小零件找它肚里的特定螺丝（层级模块）
                    if isinstance(submodule, torch.nn.Linear):  # 特么只认定那些叫 Linear (全联接）的最根本单元作为附身 LoRA 的载体因为 LoRA 全在改造线性变化
                        target_linear_modules.add(full_submodule_name)  # 把它的身份号留着
        
        target_linear_modules = list(target_linear_modules)  # 定死不可变列阵方便调用
        
        if self.is_main_process:  # 供查收校对打表看选的是否准确
            print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
            if getattr(self.lora_config, 'verbose', False):  # 会比较长，开了详细日志才刷上去
                for module_name in sorted(target_linear_modules):
                    print(f"  - {module_name}")
        
        # Create LoRA config  # PEFT (极广度微调支持库提供的类）
        adapter_type = self.lora_config.get('type', 'lora')
        if adapter_type == 'lora':  # 若当前方案选择了普通正传 LoRA
            peft_config = peft.LoraConfig(
                r=self.lora_config.get('rank', 16),  # 参数下压等级越低省显存降耗（损失性能）越大 
                lora_alpha=self.lora_config.get('alpha', None) or self.lora_config.get('rank', 16),
                lora_dropout=self.lora_config.get('dropout', 0.0),  # 过度平滑避免记住数据防过拟（加层掉点效应参数设置）
                target_modules=target_linear_modules,  # 使用由刚才捞出来的那些层层剥下来的纯 Linear 位置坐标去附身覆盖叠加！
                # task_type="FEATURE_EXTRACTION"        # Remove this; not needed for diffusion models
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')  # 如果配置超纲了如 DoRA 但系统尚未接管处理这则崩了报错
        
        # Apply LoRA to the transformer  # 开始对整个底结构灌皮套封
        lora_model = peft.get_peft_model(transformer, peft_config)

        if self.is_main_process:
            print('peft_config', peft_config)
            lora_model.print_trainable_parameters()  # 打出报告显现在 LoRA 套层下的全网当前只用微调多少数百万（可观地替代数十亿底模规模）

        return lora_model


    def _gather_lora_state_dict(self, lora_model):  # 分布式的特解辅助方法用来专门抓散落参数只提取剥离属于 LoRA 私有的部分重组保存
        "On rank-0, gather FULL_STATE_DICT, then filter only LoRA weights"
        with FSDP.state_dict_type(
            lora_model,                       # lora_model contains nested FSDP submodules
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True)  # 要求全块在内存而不是显卡上重聚！
        ):
            full = lora_model.state_dict()  # 获得带着壳的原始混合大全参数
        return get_peft_model_state_dict(lora_model, state_dict=full)  # 洗盘子一样利用库特性清洗掉底板结构只抽出 PEFT 相关的层叠记录归档留用
    
    # --------------------------------------------------------------------------------------------------------------
    # Visualization helpers  # （附属）：只负责专门造生成测试短片的配套服务管辖
    # --------------------------------------------------------------------------------------------------------------

    def _setup_visualizer(self):
        """Initialize the inference pipeline for visualization on CPU, to be moved to GPU only when needed."""
        # 设置整个视频产出流水大线的架构用于观调
        # Choose pipeline class depending on causal flag
        if 'switch' in self.config.distribution_loss:  # 根据所正在跑实验训练的网络形式类型选用应对其胃口的生图发生器接口
            self.vis_pipeline = SwitchCausalInferencePipeline(
                args=self.config,
                device=self.device,  # 让各层找到该在工作的机位内存
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae)
        else:
            self.vis_pipeline = CausalInferencePipeline(  # 普通顺拍的因果管线推求出成片法
                args=self.config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae)

        # Visualization output directory (default: <logdir>/vis)  # 把做成的胶卷储库备置好（预建在工作文件夹下的视窗目录下）
        self.vis_output_dir = os.path.join(os.path.dirname(self.output_path), "vis")
        os.makedirs(self.vis_output_dir, exist_ok=True)
        if self.config.vis_ema:
            raise NotImplementedError("Visualization with EMA is not implemented")  # 目前为了保底算力未上线推看 EMA ，故防止用户去开启了 EMA 又叫开生成而出错误踩坑提供一个拦截弹提示

    def _visualize(self):  # 前来调用上边生成的方法制造产出演示，并将那些张量视频画成实底存储并命正全名落库的主逻辑
        """Generate and save sample videos to monitor training progress."""
        if self.vis_interval <= 0 or not hasattr(self, "vis_pipeline"):  # 若未配置就不开火
            return

        # Use the fixed batch of prompts/images prepared from val_loader
        if not getattr(self, "fixed_vis_batch", None):  # 防止无米下炊因为我们上个模块只取个开头没取到可能文件损毁
            print("[Warning] No fixed validation batch available for visualization.")
            return

        if self.one_logger is not None:
            self.one_logger.on_validation_batch_start()  # 开始这批次的计票打表

        step_vis_dir = os.path.join(self.vis_output_dir, f"step_{self.step:07d}")  # 根据当前的进度建立此一步骤进度视频存档柜门夹层编号
        os.makedirs(step_vis_dir, exist_ok=True)
        batch = self.fixed_vis_batch  # 借提那些雷打不动死守此岗用作参照的提示及基座素材
        if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
            prompts = batch["prompts"]
            switch_prompts = batch["switch_prompts"]  # 带断片急切功能的管路也必跟有急切语境对照辞令提供给流段
            switch_frame_index = self._get_switch_frame_index()
        else:
            prompts = batch["prompts"]

        image = None
        if self.config.i2v and ("image" in batch):  # 如附原真图一并抽出待作底衬
            image = batch["image"]

        # Prepare model mode info for filename
        mode_info = ""
        if self.is_lora_enabled:  # 因为模型种类将影响成品好恶表现以及辨别，故在视频挂名末也加以辨认附签标清是本基或者 LoRA 身分证
            mode_info = "_lora"
            if self.is_main_process:
                print(f"Generating videos in LoRA mode (step {self.step})")
        
        for vid_len in self.vis_video_lengths:  # 既然要求了生成好多段（按秒切列），则各去跑一遍
            print(f"Generating video of length {vid_len}")
            if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
                videos = self.generate_video_with_switch(self.vis_pipeline, vid_len, prompts, switch_prompts, switch_frame_index, image=image)
            else:
                videos = self.generate_video(self.vis_pipeline, vid_len, prompts, image=image)  # 开始去生这指定长度的影片（在前面方法被唤醒解压为成形视频序列）

            # Save each sample  # 去存储每一个被制造出来的实体作品到各自对应的小框格子里
            for idx, video_np in enumerate(videos):  # 防止 Batch 里含有不唯一的片段要求产生于是切分逐项对待
                if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
                    video_name = f"step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}_switch_frame_{switch_frame_index}.mp4"
                else:
                    video_name = f"step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}.mp4"
                out_path = os.path.join(  # 把生成的前后缀配妥作为完整的储存全称指引用以给 write 放写入硬盘
                    step_vis_dir,
                    video_name,
                )
                video_tensor = torch.from_numpy(video_np.astype("uint8"))  # 使脱离 numpy 框恢复 pytorch 正经像素层张带单位 0-255 标准
                write_video(out_path, video_tensor, fps=16)  # 把它们制作为最主流接受播放广泛通用 MP4，给定标准的放映张幅帧速率

            # After saving current length videos, release related tensors to reduce peak memory  # 在换另一种长度继续制作之前立马清除上面产生的这卷昂贵胶卷省地
            del videos, video_np, video_tensor  # type: ignore
            torch.cuda.empty_cache()

        if self.one_logger is not None:
            self.one_logger.on_validation_batch_end()

        torch.cuda.empty_cache()
        import gc
        gc.collect()
