% 0. 环境与路径
dll = 'H:\Matlab\bin\BWIBridgeDLL.dll';
hdr = 'H:\Matlab\bin\BWIBridgeDLL.h';
dir = 'C:\Users\pc\Desktop\BWIS_互作用-helix-非线性特征仿真实例\注波互作用1\指定输入';

if ~libisloaded('BWIBridgeDLL')
    loadlibrary(dll,hdr);
end

p   = calllib('BWIBridgeDLL','CreateProcessor');
nPt = libpointer('int32Ptr',0);
assert(calllib('BWIBridgeDLL','ReadBWISDat',p,dir,nPt)==1,'读取失败');
N = nPt.Value;

% --- 指针缓冲区 ---
xPtr   = libpointer('doublePtr', zeros(N,1));
powPtr = libpointer('doublePtr', zeros(N,1));
gainPtr= libpointer('doublePtr', zeros(N,1));
effPtr = libpointer('doublePtr', zeros(N,1));

calllib('BWIBridgeDLL','CopyX_mm', p, xPtr,   N);
calllib('BWIBridgeDLL','CopyPower',p, powPtr, N);
calllib('BWIBridgeDLL','CopyGain', p, gainPtr,N);
calllib('BWIBridgeDLL','CopyEff',  p, effPtr ,N);

x    = xPtr.Value;
pow  = powPtr.Value;
gain = gainPtr.Value;
eff  = effPtr.Value;

% --- 绘图 ---
figure;
subplot(3,1,1); plot(x,pow,'-b');  title('输出功率曲线');xlabel('z / mm');ylabel('W');
subplot(3,1,2); plot(x,gain,'-b'); title('增益曲线');    xlabel('z / mm');ylabel('dB');
subplot(3,1,3); plot(x,eff ,'-b'); title('效率曲线');    xlabel('z / mm');ylabel('%');

% --- 清理 ---
calllib('BWIBridgeDLL','DeleteProcessor',p);
unloadlibrary('BWIBridgeDLL');
