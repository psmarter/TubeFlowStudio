% function start_import_gui()
%     % 获取屏幕大小
%     screenSize = get(0, 'ScreenSize');
%     screenWidth = screenSize(3);
%     screenHeight = screenSize(4);
% 
%     % 设置图形界面大小
%     figWidth = 800;
%     figHeight = 400;
% 
%     % 计算居中显示的坐标
%     figPosX = (screenWidth - figWidth) / 2;
%     figPosY = (screenHeight - figHeight) / 2;
% 
%     % 创建图形界面（窗口），居中显示
%     fig = uifigure('Name', '数据导入与显示', 'Position', [figPosX, figPosY, figWidth, figHeight]);
% 
%     % 创建按钮来导入数据
%     btnImport = uibutton(fig, 'push', 'Text', '导入数据', 'Position', [680, 360, 100, 30], ...
%                          'ButtonPushedFcn', @(src, event) import_data(fig));
% 
%     % 创建一个uitable用于显示数据
%     uit = uitable(fig, 'Position', [20, 60, 760, 280], ...
%                   'ColumnName', {'变量名', '数值', '最小值', '最大值', '说明'}, ...
%                   'Data', {}, 'ColumnEditable', [false true false false false]);
%     uit.Tag = 'dataTable';  % 设置标签，方便后续操作
% 
%     % 用于保存数据的临时变量
%     fig.UserData.data = [];
% 
%     % 创建一个TextArea来显示文件夹路径
%     filePathLabel = uilabel(fig, 'Text', '选择的文件夹路径：', 'Position', [20, 360, 100, 30]);
%     % filePathTextArea = uitextarea(fig, 'Position', [130, 360, 530, 30]);
%     filePathTextArea = uitextarea(fig, 'Position', [130, 360, 530, 30], 'Tag', 'filePathTextArea');
%     filePathTextArea.Editable = 'off';  % 设置为只读
% 
%     % 创建“确认”按钮
%     uibutton(fig, 'push', 'Text', '确认', 'Position', [520, 20, 100, 30], ...
%              'ButtonPushedFcn', @(src, event) confirm_action(fig));
% 
%     % 创建“取消”按钮
%     uibutton(fig, 'push', 'Text', '取消', 'Position', [640, 20, 100, 30], ...
%              'ButtonPushedFcn', @(src, event) cancel_action(fig, filePathTextArea));
% end
% 
% % 导入数据的回调函数
% function import_data(fig)
%     % 弹出文件夹选择对话框
%     folderPath = uigetdir('', '选择数据文件夹');
% 
%     if folderPath == 0  % 用户取消选择文件夹
%         return;
%     end
% 
%     % 拼接完整路径
%     iniFilePath = fullfile(folderPath, 'bwiparamtemp.ini');
% 
%     if ~isfile(iniFilePath)
%         uialert(fig, 'bwiparamtemp.ini 文件不存在', '导入错误');
%         return;
%     end
% 
%     % 调用 DLL 读取数据
%     try
%         % 创建处理器
%         dll = fullfile(pwd, 'bin', 'BWIBridgeDLL.dll');  % 使用当前工作目录
%         hdr = fullfile(pwd, 'bin', 'BWIBridgeDLL.h');
% 
%         if ~libisloaded('BWIBridgeDLL')
%             loadlibrary(dll, hdr);
%         end
% 
%         proc = calllib('BWIBridgeDLL', 'CreateProcessor');
% 
%         % 调用函数加载 INI 参数
%         calllib('BWIBridgeDLL', 'LoadParametersFromIni', proc, iniFilePath);
% 
%         % 获取参数数量
%         paramCount = calllib('BWIBridgeDLL', 'GetGlobalParameterCount', proc);
% 
%         % 获取参数的名称、数值、最小值、最大值、描述等
%         data = cell(paramCount, 5);  % 5 列：名称、数值、最小值、最大值、描述
%         for i = 0:paramCount-1
%             name = calllib('BWIBridgeDLL', 'GetParameterName', proc, i);
%             value = calllib('BWIBridgeDLL', 'GetParameterValue', proc, i);
%             minVal = calllib('BWIBridgeDLL', 'GetParameterMin', proc, i);
%             maxVal = calllib('BWIBridgeDLL', 'GetParameterMax', proc, i);
%             desc = calllib('BWIBridgeDLL', 'GetParameterDescription', proc, i);
% 
%             % 填充表格数据
%             data{i+1, 1} = name;      % 变量名
%             data{i+1, 2} = value;     % 数值 (可修改)
%             data{i+1, 3} = minVal;    % 最小值
%             data{i+1, 4} = maxVal;    % 最大值
%             data{i+1, 5} = desc;      % 说明
%         end
% 
%         % 更新表格数据
%         uit = findobj(fig, 'Tag', 'dataTable');
%         uit.Data = data;  % 显示数据
% 
%         % 保存数据到 UserData（便于后续操作）
%         fig.UserData.data = data;
% 
%         % 更新文件夹路径显示框
%         filePathTextArea = findobj(fig, 'Tag', 'filePathTextArea');
%         if ~isempty(filePathTextArea)
%             filePathTextArea.Value = folderPath;  % 显示路径
%         else
%             uialert(fig, '未找到文件夹路径文本框，无法更新路径', '更新错误');
%         end
% 
%         % 释放 DLL
%         unloadlibrary('BWIBridgeDLL');
% 
%         disp('数据导入成功');
%     catch ME
%         % 读取文件失败时的处理
%         uialert(fig, ['文件导入失败: ', ME.message], '导入错误');
%     finally
%         % 释放 DLL
%         if libisloaded('BWIBridgeDLL')
%             unloadlibrary('BWIBridgeDLL');
%         end
%     end
% end
% 
% % 确认按钮的回调
% function confirm_action(fig)
%     % 确认后，修改确认标志位为 1，表示确认
%     persistent confirmationFlag
%     confirmationFlag = 1;  % 修改为 1 表示确认
% 
%     % 关闭 GUI 窗口
%     close(fig);
% end
% 
% % 取消按钮的回调
% function cancel_action(fig, filePathTextArea)
%     % 取消后，修改确认标志位为 -1，表示取消
%     persistent confirmationFlag
%     confirmationFlag = -1;  % 修改为 -1 表示取消
% 
%     % 关闭 GUI 窗口
%     close(fig);
% end
