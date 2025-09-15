function sfun_callBWISProcess(block)
    setup(block);
end

% ─────────────────── SETUP ────────────────────
function setup(block)
    block.NumInputPorts  = 1;
    block.NumOutputPorts = 0;
    block.InputPort(1).Dimensions        = 1;
    block.InputPort(1).DatatypeID        = 0;   % double
    block.InputPort(1).Complexity        = 'Real';
    block.InputPort(1).DirectFeedthrough = true;

    block.NumDialogPrms  = 1;

    block.SampleTimes        = [0.5 0];         % 0.5 s 轮询
    block.SimStateCompliance = 'DefaultSimState';

    block.RegBlockMethod('PostPropagationSetup', @PostProp);
    block.RegBlockMethod('InitializeConditions', @InitCond);
    block.RegBlockMethod('Outputs',              @Outputs);
    block.RegBlockMethod('Terminate',            @Terminate);
end

% ─── PostPropagationSetup：声明 DWork ───
function PostProp(block)
    block.NumDworks = 1;                       % prevU
    block.Dwork(1).Name            = 'prevU';
    block.Dwork(1).Dimensions      = 1;
    block.Dwork(1).DatatypeID      = 0;
    block.Dwork(1).Complexity      = 'Real';
    block.Dwork(1).UsedAsDiscState = true;
end

% ─── 初始化 ───
function InitCond(block)
    block.Dwork(1).Data = 0;
end

% ─────────────────── Outputs ───────────────────
function Outputs(block)
    u = block.InputPort(1).Data;
    prevU = block.Dwork(1).Data;

    % 如果脉冲信号为2，则开始执行数据读取
    if u == 2 && prevU ~= 2
        % 为保险起见，检查库是否已加载
        if ~libisloaded('BWIBridgeDLL')
            fprintf('错误: BWIBridgeDLL 未被加载。仿真可能未从 sfun_callInterface 正常启动。\n');
            return;
        end

        % --- 读取数据并绘图 ---
        % dll = '.\bin\BWIBridgeDLL.dll';
        % hdr = '.\bin\BWIBridgeDLL.h';
        % dir = 'C:\Users\pc\Desktop\BWIS_互作用-helix-非线性特征仿真实例\注波互作用1\指定输入';
        dir = block.DialogPrm(1).Data;

        p   = calllib('BWIBridgeDLL', 'CreateProcessor');
        nPt = libpointer('int32Ptr', 0);
        result = calllib('BWIBridgeDLL', 'ReadBWISDat', p, dir, nPt);
        if ~result
            fprintf('读取失败')
            calllib('BWIBridgeDLL', 'DeleteProcessor', p);
            unloadlibrary('BWIBridgeDLL')
        end;

        N = nPt.Value;

        % --- 指针缓冲区 ---
        xPtr   = libpointer('doublePtr', zeros(N, 1));
        powPtr = libpointer('doublePtr', zeros(N, 1));
        gainPtr= libpointer('doublePtr', zeros(N, 1));
        effPtr = libpointer('doublePtr', zeros(N, 1));

        calllib('BWIBridgeDLL', 'CopyX_mm', p, xPtr, N);
        calllib('BWIBridgeDLL', 'CopyPower', p, powPtr, N);
        calllib('BWIBridgeDLL', 'CopyGain', p, gainPtr, N);
        calllib('BWIBridgeDLL', 'CopyEff',  p, effPtr, N);

        x    = xPtr.Value;
        pow  = powPtr.Value;
        gain = gainPtr.Value;
        eff  = effPtr.Value;

        % --- 绘图 ---
        figure;
        subplot(3, 1, 1);
        plot(x, pow, '-b');
        title('输出功率曲线');
        xlabel('z / mm');
        ylabel('W');

        subplot(3, 1, 2);
        plot(x, gain, '-b');
        title('增益曲线');
        xlabel('z / mm');
        ylabel('dB');

        subplot(3, 1, 3);
        plot(x, eff, '-b');
        title('效率曲线');
        xlabel('z / mm');
        ylabel('%');

        % 添加整体标题
        % sgtitle('行波管结果显示');

        % 修改窗口标题
        set(gcf, 'Name', '行波管结果显示', 'NumberTitle', 'off');

        % --- 清理 ---
        calllib('BWIBridgeDLL', 'DeleteProcessor', p);
        % unloadlibrary('BWIBridgeDLL');
    end

    % 保存当前脉冲值
    block.Dwork(1).Data = u;
end

% ─── Terminate：处理退出 —──
function Terminate(~)
    fprintf('[BWIS] Simulation 终止. 卸载 BWIBridgeDLL.\n');
    if libisloaded('BWIBridgeDLL')
        unloadlibrary('BWIBridgeDLL');
    end
end
