function sfun_callBWIS(block)
    setup(block);
end

% ─────────────────── SETUP ────────────────────
function setup(block)
    %— I/O 配置 —————————————————————————————————————————————
    block.NumInputPorts  = 1;
    block.NumOutputPorts = 1;
    block.InputPort(1).Dimensions        = 1;
    block.InputPort(1).DatatypeID        = 0;   % double
    block.InputPort(1).DirectFeedthrough = true;
    block.OutputPort(1).Dimensions       = 1;
    block.OutputPort(1).DatatypeID       = 0;

    block.NumDialogPrms  = 1; 

    %— SampleTime 配置 ———————————————————————
    block.SampleTimes        = [0.5 0];         % 0.5 s 轮询
    block.SimStateCompliance = 'DefaultSimState';

    %— 注册方法 —————————————————————————
    block.RegBlockMethod('PostPropagationSetup', @PostProp);
    block.RegBlockMethod('InitializeConditions', @InitCond);
    block.RegBlockMethod('Outputs',              @Outputs);
    block.RegBlockMethod('Terminate',            @Terminate);

    %— 设置仿真时间为无限，直到进程完成 —————
    set_param(bdroot(block.BlockHandle), 'StopTime', 'inf');
end

% ─── PostPropagationSetup：声明 DWork ───
function PostProp(block)
    block.NumDworks = 3;
    names = {'prevU', 'exeState', 'pid'};
    for k = 1:3
        block.Dwork(k).Name            = names{k};
        block.Dwork(k).Dimensions      = 1;
        block.Dwork(k).DatatypeID      = 0;
        block.Dwork(k).Complexity      = 'Real';
        block.Dwork(k).UsedAsDiscState = true;
    end
end

% ─── 初始化 ───
function InitCond(block)
    block.Dwork(1).Data = 0;
    block.Dwork(2).Data = 0;
    block.Dwork(3).Data = 0;
end

% ─────────────────── Outputs ───────────────────
function Outputs(block)
    u        = block.InputPort(1).Data;
    prevU    = block.Dwork(1).Data;
    exeState = block.Dwork(2).Data;
    pid      = block.Dwork(3).Data;

    %— 持久变量：进程句柄 —————————————
    persistent procObj
    if isempty(procObj), procObj = []; end

    %— 配置区：路径 —————————
    exePath   = '.\Solver\BWIS.exe';
    % paramPath = 'C:\Users\pc\Desktop\BWIS_互作用-helix-非线性特征仿真实例\注波互作用1\指定输入';
    paramPath = block.DialogPrm(1).Data; 

    %— ① Rising-edge → 启动 EXE —————————
    if u == 1 && prevU == 0 && exeState == 0
        [ok, procObj, pid, exeState] = launchExe(exePath, paramPath);
        if ok
            fprintf('[BWIS] PID=%d 已启动\n', pid);
        else
            exeState = -1;  % 启动失败
        end
    end

    %— ② 运行中 → 轮询进程是否退出 —————
    if exeState == 1
        [exeState, pid] = pollProcessExitStatus(procObj, pid);
        if exeState == 2 || exeState == -1
            fprintf('[BWIS] PID=%d 已退出\n', block.Dwork(3).Data)
            stopSimulation(block);
        end
    end

    %— ③ 输出 & 保存 —————————
    block.OutputPort(1).Data = exeState;
    block.Dwork(1).Data      = u;
    block.Dwork(2).Data      = exeState;
    block.Dwork(3).Data      = pid;
end

% ─── Terminate：处理进程退出 —──
function Terminate(~)
    persistent procObj
    if ~isempty(procObj)
        try
            terminateProcess(procObj);
        catch
            % Ignore if the process has already been terminated
        end
    end
end

% ─── helper：启动进程 ───
function [ok, proc, pid, exeState] = launchExe(exePath, paramPath)
    ok = false; proc = []; pid = 0; exeState = 0;
    try
        if exist(exePath, 'file') ~= 2
            warning('[BWIS] EXE 不存在: %s', exePath); return; end
        if exist(paramPath, 'dir') ~= 7
            warning('[BWIS] 参数路径不存在: %s', paramPath); return; end

        argStr = ['1|1|' paramPath];  % 启动参数
        proc = System.Diagnostics.Process();
        info = proc.StartInfo;
        info.FileName        = exePath;
        info.Arguments       = argStr;
        info.UseShellExecute = false;
        info.CreateNoWindow  = true;
        ok = proc.Start();
        if ok
            pid = double(proc.Id);
            exeState = 1;   % 运行中
        end
    catch ME
        warning('[BWIS] 启动 EXE 异常: %s', E.message);
    end
end

% ─── helper：轮询进程退出状态 ───
function [exeState, pid] = pollProcessExitStatus(procObj, pid)
    finished = false;
    try
        procObj.Refresh();
        finished = procObj.HasExited;
        if ~finished
            finished = procObj.WaitForExit(50);  % 等待 50 ms
        end
    catch
        finished = ~isProcAlive(pid);
    end
    
    if finished
        exitC = getExitCodeSafe(procObj);
        exeState = (exitC == 0) * 2 + (exitC ~= 0) * (-1);
        pid = 0;
    else
        exeState = 1; % Running
    end
end

% ─── helper：检查 PID 是否仍存活 ───
function alive = isProcAlive(pid)
    alive = false;
    if pid <= 0, return; end
    try
        pchk = System.Diagnostics.Process.GetProcessById(pid);
        alive = ~pchk.HasExited;
    catch
        % Process not found
    end
end

% ─── helper：读取 ExitCode ───
function code = getExitCodeSafe(procObj)
    code = -1;
    try code = procObj.ExitCode; catch, end
end

% ─── helper：终止进程 ───
function terminateProcess(procObj)
    if ~procObj.HasExited
        procObj.Kill();
    end
    procObj.Dispose();
end

% ─── helper：停止仿真 ───
function stopSimulation(block)
    set_param(bdroot(block.BlockHandle), 'SimulationCommand', 'stop');
end
