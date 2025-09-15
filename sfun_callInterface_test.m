function varargout = sfun_callInterface(varargin)
    % 主函数入口，支持多种调用方式
    if nargin > 0 && ischar(varargin{1})
        % 如果是通过名称调用，执行相应的功能
        switch varargin{1}
            case 'open_gui'
                % 双击模块时调用
                blockHandle = varargin{2};
                start_import_gui(blockHandle);
            otherwise
                % 默认情况，作为 S-Function 使用
                if nargout > 0
                    [varargout{1:nargout}] = feval(varargin{:});
                else
                    feval(varargin{:});
                end
        end
    else
        % 作为 S-Function 使用
        if nargout > 0
            [varargout{1:nargout}] = block_api(varargin{:});
        else
            block_api(varargin{:});
        end
    end
end

% S-Function 的主要接口函数
function varargout = block_api(block, varargin)
    % 根据不同的调用模式执行相应的功能
    action = varargin{1};
    switch action
        case 'setup'
            setup(block);
        case 'postpropagation'
            postPropagationSetup(block);
        case 'initialize'
            initializeConditions(block);
        case 'outputs'
            outputs(block);
        case 'terminate'
            terminate(block);
        otherwise
            % 未知操作
            if nargout > 0
                varargout{1} = [];
            end
    end
end

% ─────────────────── SETUP ────────────────────
function setup(block)
    % I/O 配置（没有输入端口）
    block.NumInputPorts  = 0;
    block.NumOutputPorts = 1;
    block.OutputPort(1).Dimensions       = 1;
    block.OutputPort(1).DatatypeID       = 0;  % 输出一个常数，作为标志位

    % 明确设置输出端口的采样模式（设置为样本数据）
    block.OutputPort(1).SamplingMode = 'Sample';

    % SampleTime 配置
    block.SampleTimes        = [0 0];         % 立即执行
    block.SimStateCompliance = 'DefaultSimState';

    % 注册方法
    block.RegBlockMethod('PostPropagationSetup', @postPropagationSetup);
    block.RegBlockMethod('InitializeConditions', @initializeConditions);
    block.RegBlockMethod('Outputs', @outputs);
    block.RegBlockMethod('Terminate', @terminate);
    
    % 设置模块的 OpenFcn 回调
    set_param(block.BlockHandle, 'OpenFcn', 'sfun_callInterface(''open_gui'', gcbh)');
end

% ─── PostPropagationSetup：声明 DWork ───
function postPropagationSetup(block)
    % 确保先设置 DWork 的数量
    block.NumDworks = 2;
    
    % Dwork 1: 用于存储 GUI 是否已启动的标志
    block.Dwork(1).Name            = 'guiLaunched';
    block.Dwork(1).Dimensions      = 1;
    block.Dwork(1).DatatypeID      = 0;
    block.Dwork(1).Complexity      = 'Real';
    block.Dwork(1).UsedAsDiscState = true;
    
    % Dwork 2: 用于存储确认标志
    block.Dwork(2).Name            = 'confirmationFlag';
    block.Dwork(2).Dimensions      = 1;
    block.Dwork(2).DatatypeID      = 0;
    block.Dwork(2).Complexity      = 'Real';
    block.Dwork(2).UsedAsDiscState = true;
end

% ─── InitializeConditions：初始化 DWork ───
function initializeConditions(block)
    % 确保 DWork 已正确初始化后再访问
    if block.NumDworks >= 1
        block.Dwork(1).Data = 0;  % GUI 未启动
    end
    
    if block.NumDworks >= 2
        block.Dwork(2).Data = 0;  % 确认标志为 0
    end
end

% ─── Outputs ───
function outputs(block)
    % 确保 DWork 已正确初始化后再访问
    if block.NumDworks < 2
        % 如果 DWork 未正确初始化，设置默认值
        confirmationFlag = 0;
    else
        % 获取当前状态
        guiLaunched = block.Dwork(1).Data;
        confirmationFlag = block.Dwork(2).Data;
        
        % 如果 GUI 尚未启动，则启动 GUI
        if guiLaunched == 0
            start_import_gui(block.BlockHandle);  % 传递块句柄给 GUI
            block.Dwork(1).Data = 1;  % 设置 GUI 已启动标志
        end
        
        % 尝试从基础工作空间获取确认状态
        try
            confirmationFlag = evalin('base', 'confirmationFlag');
            block.Dwork(2).Data = confirmationFlag;
        catch
            % 如果变量不存在，保持当前值
        end
    end
    
    % 输出确认标志
    block.OutputPort(1).Data = confirmationFlag;
    
    % 如果确认标志是 1，自动运行仿真
    if confirmationFlag == 1
        % 自动开始仿真
        set_param(bdroot(block.BlockHandle), 'SimulationCommand', 'start');
    end

    % 如果确认标志是 -1，表示取消，停止仿真
    if confirmationFlag == -1
        try
            % 检查仿真是否正在运行，若正在运行则停止
            simStatus = get_param(bdroot(block.BlockHandle), 'SimulationStatus');
            if strcmp(simStatus, 'running')
                % 如果仿真正在运行，取消仿真
                set_param(bdroot(block.BlockHandle), 'SimulationCommand', 'stop');
                disp('仿真已取消');
            end
        catch
            % 如果没有仿真，或其他错误，则跳过
        end
    end
end

% ─── Terminate：处理进程退出 ───
function terminate(~)
    % 清理基础工作空间中的标志变量
    try
        evalin('base', 'clear confirmationFlag');
    catch
    end
end

% GUI 启动函数
function start_import_gui(blockHandle)
    % 检查是否已经有打开的 GUI，避免重复打开
    fig = findall(0, 'Type', 'Figure', 'Name', '数据导入与显示');
    if ~isempty(fig)
        figure(fig);  % 将现有窗口置于前台
        return;
    end
    
    % 获取屏幕大小
    screenSize = get(0, 'ScreenSize');
    screenWidth = screenSize(3);
    screenHeight = screenSize(4);
    
    % 设置图形界面大小
    figWidth = 800;
    figHeight = 400;

    % 计算居中显示的坐标
    figPosX = (screenWidth - figWidth) / 2;
    figPosY = (screenHeight - figHeight) / 2;

    % 创建图形界面（窗口），居中显示
    fig = uifigure('Name', '数据导入与显示', 'Position', [figPosX, figPosY, figWidth, figHeight], ...
                  'CloseRequestFcn', @(src, event) close_gui(src));

    % 存储块句柄在图形对象的 UserData 中
    fig.UserData.blockHandle = blockHandle;
    fig.UserData.data = [];  % 初始化数据存储

    % 创建按钮来导入数据
    btnImport = uibutton(fig, 'push', 'Text', '导入数据', 'Position', [680, 360, 100, 30], ...
                         'ButtonPushedFcn', @(src, event) import_data(fig));

    % 创建一个uitable用于显示数据
    uit = uitable(fig, 'Position', [20, 60, 760, 280], ...
                  'ColumnName', {'变量名', '数值', '最小值', '最大值', '说明'}, ...
                  'Data', {}, 'ColumnEditable', [false true false false false]);
    uit.Tag = 'dataTable';  % 设置标签，方便后续操作

    % 创建一个TextArea来显示文件夹路径
    filePathLabel = uilabel(fig, 'Text', '选择的文件夹路径：', 'Position', [20, 360, 100, 30]);
    filePathTextArea = uitextarea(fig, 'Position', [130, 360, 530, 30], 'Tag', 'filePathTextArea');
    filePathTextArea.Editable = 'off';  % 设置为只读

    % 创建"确认"按钮
    uibutton(fig, 'push', 'Text', '确认', 'Position', [520, 20, 100, 30], ...
             'ButtonPushedFcn', @(src, event) confirm_action(fig));

    % 创建"取消"按钮
    uibutton(fig, 'push', 'Text', '取消', 'Position', [640, 20, 100, 30], ...
             'ButtonPushedFcn', @(src, event) cancel_action(fig));
end

% GUI 关闭请求函数
function close_gui(fig)
    % 检查是否有未确认的数据
    if isfield(fig.UserData, 'data') && ~isempty(fig.UserData.data)
        % 询问用户是否要保存更改
        selection = uiconfirm(fig, '您有未确认的更改。是否要保存？', '确认关闭', ...
                             'Options', {'保存并关闭', '不保存关闭', '取消'}, ...
                             'DefaultOption', 1, 'CancelOption', 3);
        
        switch selection
            case '保存并关闭'
                confirm_action(fig);
            case '不保存关闭'
                cancel_action(fig);
            case '取消'
                return;  % 不关闭窗口
        end
    else
        % 没有数据，直接关闭
        delete(fig);
    end
end

% 导入数据的回调函数
function import_data(fig)
    % 弹出文件夹选择对话框
    folderPath = uigetdir('', '选择数据文件夹');
    
    if folderPath == 0  % 用户取消选择文件夹
        return;
    end

    % 拼接完整路径
    iniFilePath = fullfile(folderPath, 'bwiparamtemp.ini');
    
    if ~isfile(iniFilePath)
        uialert(fig, 'bwiparamtemp.ini 文件不存在', '导入错误');
        return;
    end
    
    % 调用 DLL 读取数据
    try
        % 创建处理器
        dll = fullfile(pwd, 'bin', 'BWIBridgeDLL.dll');
        hdr = fullfile(pwd, 'bin', 'BWIBridgeDLL.h');

        if ~libisloaded('BWIBridgeDLL')
            loadlibrary(dll, hdr);
        end

        proc = calllib('BWIBridgeDLL', 'CreateProcessor');
        
        % 调用函数加载 INI 参数
        calllib('BWIBridgeDLL', 'LoadParametersFromIni', proc, iniFilePath);
        
        % 获取参数数量
        paramCount = calllib('BWIBridgeDLL', 'GetGlobalParameterCount', proc);
        
        % 获取参数的名称、数值、最小值、最大值、描述等
        data = cell(paramCount, 5);  % 5 列：名称、数值、最小值、最大值、描述
        for i = 0:paramCount-1
            name = calllib('BWIBridgeDLL', 'GetParameterName', proc, i);
            value = calllib('BWIBridgeDLL', 'GetParameterValue', proc, i);
            minVal = calllib('BWIBridgeDLL', 'GetParameterMin', proc, i);
            maxVal = calllib('BWIBridgeDLL', 'GetParameterMax', proc, i);
            desc = calllib('BWIBridgeDLL', 'GetParameterDescription', proc, i);
            
            % 填充表格数据
            data{i+1, 1} = name;
            data{i+1, 2} = value;
            data{i+1, 3} = minVal;
            data{i+1, 4} = maxVal;
            data{i+1, 5} = desc;
        end
        
        % 更新表格数据
        uit = findobj(fig, 'Tag', 'dataTable');
        uit.Data = data;
        
        % 保存数据到 UserData
        fig.UserData.data = data;

        % 更新文件夹路径显示框
        filePathTextArea = findobj(fig, 'Tag', 'filePathTextArea');
        if ~isempty(filePathTextArea)
            filePathTextArea.Value = folderPath;
        end
        
        % 释放 DLL
        unloadlibrary('BWIBridgeDLL');
        
        disp('数据导入成功');
    catch ME
        % 读取文件失败时的处理
        uialert(fig, ['文件导入失败: ', ME.message], '导入错误');
    finally
        % 释放 DLL
        if libisloaded('BWIBridgeDLL')
            unloadlibrary('BWIBridgeDLL');
        end
    end
end

% 确认按钮的回调
function confirm_action(fig)
    % 在基础工作空间中设置确认标志为 1
    assignin('base', 'confirmationFlag', 1);
    
    % 保存任何修改过的数据（如果有）
    if isfield(fig.UserData, 'data') && ~isempty(fig.UserData.data)
        % 这里可以添加代码保存修改后的数据
        uit = findobj(fig, 'Tag', 'dataTable');
        modifiedData = uit.Data;
        
        % 将修改后的数据保存到基础工作空间或其他地方
        assignin('base', 'importedData', modifiedData);
        disp('数据已保存');
    end
    
    % 关闭 GUI 窗口
    delete(fig);
end

% 取消按钮的回调
function cancel_action(fig)
    % 在基础工作空间中设置确认标志为 -1
    assignin('base', 'confirmationFlag', -1);
    
    % 关闭 GUI 窗口
    delete(fig);
end