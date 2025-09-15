function sfun_callGunProcess(block)
    setup(block);
end

% ─────────────────── SETUP ────────────────────
function setup(block)
    % 端口与 sfun_callBWISProcess 相同：1 输入，0 输出
    block.NumInputPorts  = 1;
    block.NumOutputPorts = 0;

    block.InputPort(1).Dimensions        = 1;
    block.InputPort(1).DatatypeID        = 0;   % double
    block.InputPort(1).Complexity        = 'Real';
    block.InputPort(1).DirectFeedthrough = true;

    % 一个对话参数：电子枪工程路径（string）
    % block.NumDialogPrms  = 1;

    % Sample time
    block.SampleTimes        = [0.5 0];         % 0.5 s 轮询
    block.SimStateCompliance = 'DefaultSimState';

    % 注册方法
    block.RegBlockMethod('PostPropagationSetup', @PostProp);
    block.RegBlockMethod('InitializeConditions', @InitCond);
    block.RegBlockMethod('Outputs',              @Outputs);
    block.RegBlockMethod('Terminate',            @Terminate);
end

% ─── PostPropagationSetup：声明 DWork ───
function PostProp(block)
    % 仅保存上一次输入值 prevU
    block.NumDworks = 1;
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
    u     = block.InputPort(1).Data;
    prevU = block.Dwork(1).Data;
    
    % ↑ 上升沿触发：当输入从非 2 变为 2 时执行
    if u == 2 && prevU ~= 2
        %—— 配置区：DLL 与路径 ——%
        dllName    = 'BWIBridgeDLL';
        dllPath    = '.\bin\BWIBridgeDLL.dll';
        headerPath = '.\bin\BWIBridgeDLL.h'; % 确保头文件里有新函数的声明
        exePath    = '.\Solver\EOS.exe';
        projectPath = 'C:\Users\pc\Desktop\EOS_磁控注入枪-helix\电子枪1';
        
        %—— 加载库 ——%
        if ~libisloaded(dllName)
            loadlibrary(dllPath, headerPath);
            fprintf('[GUN DLL] 库 "%s" 加载成功。\n', dllName);
        end
        
        pGun = [];
        
        try
            % 创建电子枪处理器
            pGun = calllib(dllName, 'CreateGunProcessor', projectPath, exePath);
            if isempty(pGun) || (isa(pGun,'lib.pointer') && pGun.isNull)
                error('[GUN DLL] CreateGunProcessor 失败（空指针）。');
            end
            fprintf('[GUN DLL] Gun 处理器已创建。\n');
            
            % 读取 Result.ini -> m_vEOSGunResult
            okRead = calllib(dllName, 'ReadGunResultIni', pGun);
            if ~okRead
                error('[GUN DLL] ReadGunResultIni 失败或无结果。');
            end
            
            
            % 1. 在 MATLAB 中预分配一个足够大的缓冲区
            BUFFER_SIZE = 4096; % 假设 4KB 足够大，您可以根据实际情况调整
            jsonBuffer = char(zeros(1, BUFFER_SIZE));

            % 2. 调用新的 C++ 接口，将缓冲区传递过去
            % 注意参数：pGun, 缓冲区, 缓冲区大小
            charsWritten = calllib(dllName, 'GetGunResultsAsJson', pGun, jsonBuffer, BUFFER_SIZE);
            
            % 3. 检查返回值
            if charsWritten < 0
                error('[GUN DLL] GetGunResultsAsJson 失败。错误码: %d。可能缓冲区太小。', charsWritten);
            end
            
            % 4. 从缓冲区中提取有效的 JSON 字符串
            % 截取从第一个字符到实际写入长度的子字符串
            jsonString = jsonBuffer(1:charsWritten);
            
            % 5. 使用 MATLAB 内置函数解析 JSON
            resultData = jsondecode(jsonString);
            
            % ==========================================================
            
            fprintf('[GUN DLL] 成功解析 %d 条键值对。\n', numel(resultData));
            if ~isempty(resultData)
                % 将解析出的结构体数组直接转换为 Table
                T = struct2table(resultData);
                
                % 输出成表格
                disp(T);
                % 可视化成 uitable
                f = figure('Name','电子枪 Result.ini 键值', 'NumberTitle','off');
                uitable('Parent', f, ...
                        'Data', T{:,:}, ...
                        'ColumnName', T.Properties.VariableNames, ...
                        'Units','normalized', ...
                        'Position',[0 0 1 1]);
            else
                fprintf('[GUN DLL] 结果为空。\n');
            end
            
        catch ME
            % 捕获到错误后，在这里打印错误信息
            fprintf(2, '错误: %s\n', ME.message);
            if ~isempty(ME.stack)
                fprintf(2, '错误位置: %s, 第 %d 行\n', ME.stack(1).name, ME.stack(1).line);
            end
        end
        
        %—— 清理 ——%
        % 现在不再需要释放 JSON 字符串了
        
        % 删除处理器实例
        if ~isempty(pGun) && ~(isa(pGun,'lib.pointer') && pGun.isNull)
            calllib(dllName, 'DeleteGunProcessor', pGun);
            fprintf('[GUN DLL] Gun 处理器已删除。\n');
        end
    end
    
    % 保存当前输入值
    block.Dwork(1).Data = u;
end

% ─── Terminate：退出时卸载库 —──
function Terminate(~)
    if libisloaded('BWIBridgeDLL')
        fprintf('[GUN DLL] 卸载 BWIBridgeDLL。\n');
        unloadlibrary('BWIBridgeDLL');
    end
end
