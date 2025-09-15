#pragma once

#ifdef BWIBRIDGEDLL_EXPORTS
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif
    /* BWIS */
	// ����������
    EXPORT_API void* CreateProcessor();
    EXPORT_API void DeleteProcessor(void* p);
    
    // �����ȡ
    EXPORT_API int  ReadBWISDat(void* p, const char* folder, int* pNumPts);
    EXPORT_API int   GetPointCount(void* p);                      /* �����ܵ���         */
    EXPORT_API int   CopyX_mm(void* p, double* out, int max);  /* �������� (mm)     */
    EXPORT_API int   CopyPower(void* p, double* out, int max);  /* ������� (W)      */
    EXPORT_API int   CopyGain(void* p, double* out, int max);  /* ���� (dB)         */
    EXPORT_API int   CopyEff(void* p, double* out, int max);  /* Ч�� (%)          */

    // ��������
    EXPORT_API void LoadParametersFromIni(void* p, const char* filename);
    EXPORT_API int GetGlobalParameterCount(void* p);
    EXPORT_API const char* GetParameterName(void* p, int index);
    EXPORT_API double GetParameterValue(void* p, int index);
    EXPORT_API double GetParameterMin(void* p, int index);
    EXPORT_API double GetParameterMax(void* p, int index);
    EXPORT_API const char* GetParameterDescription(void* p, int index);
    EXPORT_API bool RewriteBWIIni_Param(void* p, const char* paramName, double newValue, const char* projectPath);


    /* GUN */
    EXPORT_API void* CreateGunProcessor(const char* projectPath, const char* execPath);
    EXPORT_API void DeleteGunProcessor(void* p);

	// ���з���
    EXPORT_API bool RunGunSimulation(void* p);
	EXPORT_API bool ReadGunResultIni(void* p);      // ��ȡ���
    EXPORT_API int GetGunResultsAsJson(void* p, char* buffer, int bufferSize);

#ifdef __cplusplus
}
#endif