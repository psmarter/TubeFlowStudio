function sfun_callGun(block)
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

    % block.NumDialogPrms  = 1; 

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
        % 确保库在仿真开始时已卸载
    if libisloaded('BWIBridgeDLL')
        unloadlibrary('BWIBridgeDLL');
    end
end

% ─────────────────── Outputs ───────────────────
function Outputs(block)
    u        = block.InputPort(1).Data;
    prevU    = block.Dwork(1).Data;
    exeState = block.Dwork(2).Data;
    pid      = 0; % PID 不再通过此方法管理

    %— 配置区：路径 —————————
    dllName     = 'BWIBridgeDLL';
    dllPath     = '.\bin\BWIBridgeDLL.dll'; 
    headerPath  = '.\bin\BWIBridgeDLL.h';   
    exePath     = '.\Solver\EOS.exe'; % 仍需传递给 DLL
    % paramPath   = block.DialogPrm(1).Data; 
    paramPath = 'C:\Users\pc\Desktop\EOS_磁控注入枪-helix\电子枪1';

    %— ① Rising-edge → 通过 DLL 启动仿真 —————————
    if u == 1 && prevU ~= 1 && exeState == 0
        fprintf('[GUN DLL] 检测到上升沿。正在启动仿真...\n');
        
        % 加载库
        if ~libisloaded(dllName)
            loadlibrary(dllPath, headerPath);
            fprintf('[GUN DLL] 库 "%s" 加载成功。\n', dllName);
        end
        
        % 创建 Gun 处理器
        pGun = calllib(dllName, 'CreateGunProcessor', paramPath, exePath);
        
        if isempty(pGun) || (isa(pGun, 'lib.pointer') && pGun.isNull)
            warning('[GUN DLL] 创建 Gun 处理器失败。');
            exeState = -1; % 启动失败
        else
            fprintf('[GUN DLL] Gun 处理器已创建。正在运行仿真...\n');
            % 运行仿真 (此为阻塞调用)
            success = calllib(dllName, 'RunGunSimulation', pGun);
            
            % 根据结果设置状态
            if success
                fprintf('[GUN DLL] 仿真成功完成。\n');
                exeState = 2; % 成功完成
            else
                fprintf('[GUN DLL] 仿真失败。\n');
                exeState = -1; % 失败
            end
            
            % 清理
            calllib(dllName, 'DeleteGunProcessor', pGun);
            fprintf('[GUN DLL] Gun 处理器已删除。\n');
        end
        
        % 任务完成，停止仿真
        stopSimulation(block);
    end

    %— ② 运行中 → 轮询进程是否退出 (此部分不再需要，因为 DLL 调用是阻塞的) —————

    %— ③ 输出 & 保存 —————————
    block.OutputPort(1).Data = exeState;
    block.Dwork(1).Data      = u;
    block.Dwork(2).Data      = exeState;
    block.Dwork(3).Data      = pid;
end

% ─── Terminate：处理进程退出 —──
function Terminate(~)
    % Terminate 主要负责卸载库
    if libisloaded('BWIBridgeDLL')
        fprintf('[GUN DLL] 仿真终止。正在卸载 BWIBridgeDLL。\n');
        unloadlibrary('BWIBridgeDLL');
    end
end

% ─── helper：停止仿真 ───
function stopSimulation(block)
    set_param(bdroot(block.BlockHandle), 'SimulationCommand', 'stop');
end