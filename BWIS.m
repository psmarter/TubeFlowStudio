function BWIS(block)
% Level-2 MATLAB S-Function：检测 0→1 边沿并调用 BWIS.exe（经 PowerShell）
    setup(block);
end

%-----------------------------------------------------------%
function setup(block)
    %% 1. 端口
    block.NumInputPorts  = 1;
    block.NumOutputPorts = 1;

    block.SetPreCompInpPortInfoToDynamic;
    block.SetPreCompOutPortInfoToDynamic;

    block.InputPort(1).Dimensions        = 1;
    block.InputPort(1).DatatypeID        = 0;  % double
    block.InputPort(1).Complexity        = 'Real';
    block.InputPort(1).DirectFeedthrough = true;

    block.OutputPort(1).Dimensions = 1;
    block.OutputPort(1).DatatypeID = 0;
    block.OutputPort(1).Complexity = 'Real';

    %% 2. 采样时间
    block.SampleTimes = [1 0];

    %% 3. 合规
    block.SimStateCompliance = 'DefaultSimState';

    %% 4. 回调
    block.RegBlockMethod('PostPropagationSetup', @PostPropSetup);
    block.RegBlockMethod('InitializeConditions', @InitConditions);
    block.RegBlockMethod('Outputs',              @Output);
end

%-----------------------------------------------------------%
function PostPropSetup(block)
    block.NumDworks = 1;
    block.Dwork(1).Name            = 'prevInput';
    block.Dwork(1).Dimensions      = 1;
    block.Dwork(1).DatatypeID      = 0;  % double
    block.Dwork(1).Complexity      = 'Real';
    block.Dwork(1).UsedAsDiscState = true;
end

%-----------------------------------------------------------%
function InitConditions(block)
    block.Dwork(1).Data = 0;
end

%-----------------------------------------------------------%
function Output(block)
    u    = block.InputPort(1).Data;  % 上一次输入
    prev = block.Dwork(1).Data;

    if u == 1 && prev == 0
        %% 用户设定路径
        exePath   = 'I:\DM202409\VexModelViewer\bin\NT_VC19_64_DLLD\OldSolver\BWIS.exe';
        paramPath = 'C:\Users\pc\Desktop\BWIS_互作用-helix-非线性特征仿真实例\注波互作用1\指定输入';

        %% 合法性检查
        if exist(exePath, 'file') ~= 2
            warning('[BWIS] EXE 不存在: %s', exePath);
            block.OutputPort(1).Data = -2;
            block.Dwork(1).Data = u;
            return;
        elseif exist(paramPath, 'dir') ~= 7
            warning('[BWIS] 参数路径不存在: %s', paramPath);
            block.OutputPort(1).Data = -3;
            block.Dwork(1).Data = u;
            return;
        end

        %% 调用 EXE（PowerShell 方式）
        try
            % 构造参数字符串
            argStr = ['1|1|' paramPath];

            % 将路径中 \ 转义为 \\
            exePathEscaped = replace(exePath, '\', '\\');
            argStrEscaped  = replace(argStr,  '\', '\\');

            % 构造命令
            powershellCmd = sprintf( ...
                'powershell -command "& ''%s'' ''%s''"', ...
                exePathEscaped, argStrEscaped);

            disp(['[BWIS] 执行: ' powershellCmd]);

            status = system(powershellCmd);

            if status == 0
                block.OutputPort(1).Data = 1;   % 成功
            else
                block.OutputPort(1).Data = -1;  % EXE 执行失败
            end
        catch ME
            warning('[BWIS] 执行异常: %s', getReport(ME));
            block.OutputPort(1).Data = -4;
        end
    else
        block.OutputPort(1).Data = 0;  % 未触发
    end

    block.Dwork(1).Data = u;           % 更新状态
end

