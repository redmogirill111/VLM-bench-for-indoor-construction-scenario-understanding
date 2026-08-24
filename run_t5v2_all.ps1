# T5v2 全量重测编排脚本：3 个 conda 环境、9 个模型，快→慢顺序
# 每阶段崩溃自动重试（最多 3 次），推理脚本内部按 (模型, 片段) 断点续跑。
# 用法:  powershell -NoProfile -ExecutionPolicy Bypass -File run_t5v2_all.ps1
# 日志:  H:\benchmark\logs\t5v2\

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"

$pyBench = "D:\miniconda\envs\vlm_bench\python.exe"
$pyEagle = "D:\miniconda\envs\eagle25\python.exe"
$pyVlm3  = "D:\miniconda\envs\videollama3_env\python.exe"
$pyMolmo = "D:\miniconda\envs\molmo2\python.exe"   # Molmo2 需 torch>=2.6
$script  = "H:\benchmark\scripts\run_t5v2_inference.py"
$logDir  = "H:\benchmark\logs\t5v2"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 快→慢（按主实验实测单次调用延迟排序），最快模型先出全量结果验证管线
# InternVL2.5/3、MiniCPM 用 videollama3_env（transformers 4.46.3，已冒烟验证）：
#   vlm_bench 的 5.7.0 与 generate 不兼容；eagle25 的 4.57.1 加载 InternVL tokenizer 返回 bool
$stages = @(
    @{ py = $pyBench; models = "Qwen2.5-VL-7B-Instruct";              log = "stage01_qwen25.log" },
    @{ py = $pyVlm3;  models = "VideoLLaMA3-7B";                       log = "stage02_videollama3.log" },
    @{ py = $pyVlm3;  models = "InternVL2.5-8B";                       log = "stage03_internvl25.log" },
    @{ py = $pyBench; models = "Ovis2-8B";                             log = "stage04_ovis2.log" },
    @{ py = $pyVlm3;  models = "InternVL3-8B";                         log = "stage05_internvl3.log" },
    @{ py = $pyBench; models = "Qwen3-VL-8B-Instruct";                 log = "stage06_qwen3.log" },
    @{ py = $pyVlm3;  models = "MiniCPM-V-2.6-8B";                     log = "stage07_minicpm.log" },
    @{ py = $pyEagle; models = "Eagle2.5-8B";                          log = "stage08_eagle25.log" },
    @{ py = $pyMolmo; models = "Molmo2-8B";                            log = "stage09_molmo2.log" }
)

Write-Host ("=" * 72)
Write-Host "T5v2 full re-test orchestrator | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ("=" * 72)

foreach ($s in $stages) {
    $logFile = Join-Path $logDir $s.log
    $completed = "H:\benchmark\output_t5v2\$($s.models)\_COMPLETED.json"
    if (Test-Path $completed) {
        Write-Host "[SKIP] $($s.models) already completed"
        continue
    }
    for ($try = 1; $try -le 3; $try++) {
        Write-Host "`n=== $($s.models) attempt $try | $(Get-Date -Format 'HH:mm:ss') ==="
        & $s.py $script --models $s.models *>> $logFile
        if ($LASTEXITCODE -eq 0) { break }
        Write-Host "[RETRY] $($s.models) exited with $LASTEXITCODE, retrying in 30s..."
        Start-Sleep -Seconds 30
    }
}

Write-Host "`nAll stages finished | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
