# **2023 데이터청년캠퍼스 : Deep Sleep**
![image](https://github.com/lim4373/deepsleep/assets/114973162/633f18d3-e740-427c-bdfe-efc84c20265d)
![image](https://github.com/lim4373/deepsleep/assets/114973162/277f7be0-9c5c-4f50-afdc-0c885f357009)

**Deep Sleep** 은 스마트워치 데이터를 이용해 수면 무호흡증을 예측하는 서비스입니다. AHI 지수를 예측하여 수면무호흡증의 중증도를 알려줍니다.
<br><br>
<img src="https://github.com/alacori/deepsleep/assets/70925118/f3d152f1-18ab-427c-9d26-ff98e0c8a5e2">
<img src="https://github.com/alacori/deepsleep/assets/70925118/95459903-1af4-46a6-92a4-ea82626c86bb">



<br><br>


## 🛌 **주제선정 및 프로젝트 배경**
![](https://github.com/alacori/deepsleep/assets/70925118/e5ec1c1b-9d1e-4e60-bca0-8bf8e150db71)
![](https://github.com/alacori/deepsleep/assets/54527982/90ef978a-1310-4809-bdd7-8f117faa26da)
![](https://github.com/alacori/deepsleep/assets/54527982/27d7412d-522c-47f3-92bf-2c811596c94d)



수면 무호흡증은 수면 중 호흡이 불규칙하거나 멈추는 장애로, 발생 빈도가 증가하고 위험성이 커지고 있습니다. 이로 인해 뇌졸중, 심근경색, 사망 등 다양한 위험이 증가하며 수면의 질 저하 및 주간 피로도 초래됩니다. 현재 진단은 병원에서의 복잡한 수면다원검사가 필요하지만, 이는 비용과 시간적 부담이 큽니다. 또한 1인 가구 증가로 수면 무호흡증 자각이 줄어드는 상황입니다. 따라서 스마트워치를 이용한 간편한 정보 수집으로 수면 무호흡증을 조기 예측하고 집에서도 관리할 수 있는 서비스를 개발하여, 치료와 삶의 질 개선에 기여하고자 하는 목적에서 "스마트워치를 이용한 수면 무호흡증 예측 서비스" 주제를 선정하게 되었습니다.
<br>
<br>

## 📈 **PSG 분석**
<img src='https://github.com/alacori/deepsleep/assets/54527982/90c33fe3-5ffd-4fdc-b964-9a82564751f0'>

수면다원검사(PSG) 분석에서는 스마트워치로 측정 가능한 변수들을 기반으로 AHI와 연관성이 있는 변수를 추출해 분석을 진행하고, 최적의 모델을 찾아 수면 무호흡증의 여부와 더불어 중증도를 쉽게 알 수 있는 웹 서비스를 제공합니다.
<br>
<br>

## 🫀 **ECG 분석**
<img src='https://github.com/alacori/deepsleep/assets/54527982/adf1bf0c-3b64-44b9-9110-f52c79224050'>


ECG 분석에서는 심전도의 심장 박동과 관련된 다양한 정보를 추출합니다. 그 중에서도 심박변이율(HRV)은 심장 박동의 불규칙성을 나타내는 지표로 활용됩니다. 이렇게 추출된 HRV 정보를 기반으로 수면 무호흡증을 예측하는 모델을 구축합니다. 이 모델은 수면과 호흡의 상관관계를 분석하여, 수면 중 무호흡증 발생 가능성을 예측하는 데 도움을 줍니다. 따라서 ECG 분석은 수면 무호흡증 예측 분야에서 매우 중요한 역할을 수행합니다.
<br>
<br>

## 🎯 **기대효과**

![](https://github.com/alacori/deepsleep/assets/54527982/4ef30424-8ade-4c90-96cb-e20d1298fc76)

먼저 스마트워치를 활용하여 간편하고 경제적인 방법으로 수면 무호흡증 예측이 가능합니다. 이는 복잡하고 비용이 많이 드는 수면다원검사(PSG) 대신 더 편리하게 수면 건강을 모니터링 할 수 있을 것입니다.
또한, 스마트워치와 웹 서비스를 통한 AHI 지수 예측과 위험도 정보 제공으로 수면 무호흡증의 조기 발견이 가능할 것입니다. 조기에 문제를 파악하고 조치함으로써 심각한 합병증을 피하고 건강한 수면을 유지할 수 있습니다.
마지막으로 스마트워치와 웹 서비스를 통한 AHI 모니터링으로 수면 무호흡증의 변화를 지속적으로 추적할 수 있습니다. 이를 통해 개인의 치료 효과를 평가하고 개선 여부를 확인할 수 있을 것입니다.
<br>
<br>

## 🛠 **Project Architecture**

<img src="https://github.com/alacori/deepsleep/assets/98305116/0524bcdb-0537-4aa6-9445-ae286c95cf87" width=900 height=600>




<br>


## 🖥️ **Feature Screen shots**

|<img src="https://github.com/alacori/deepsleep/assets/98305116/8cf96749-49ec-4c1b-a6b0-060ea692098e">|<img src="https://github.com/alacori/deepsleep/assets/98305116/c8013905-d6ca-444c-b247-95ba1586306a">|
|------|------|
|<div align="center">**PSG analysis**</div>|<div align="center">**ECG analysis**</div>|

<br> <br>



## 👩‍💻 **Contributors**

### **Team 3** : Deep Sleep

|**정시은**|**주윤나**|**한준호**|**박유진**|**신예진**|**최성림**|
|---|---|---|---|---|---|
|<img src="https://github.com/alacori/deepsleep/assets/70925118/97c473e4-bb82-4492-a009-1177484ba204">|<img src="https://github.com/alacori/deepsleep/assets/70925118/e8dfdf5c-c0f6-49b6-a8d9-6aa847198365">|<img src="https://github.com/alacori/deepsleep/assets/70925118/3bdba795-02bb-451a-bf21-52bf5a3eb688">|<img src="https://github.com/alacori/deepsleep/assets/98305116/a011e2b8-af4a-402f-a142-7fb70e7a1659">|<img src="https://github.com/alacori/deepsleep/assets/98305116/edac53b2-267d-4e76-843c-6944b1116d9b">|<img src="https://github.com/alacori/deepsleep/assets/98305116/228b81ff-2e91-41d9-abf5-feb590607038">|
|<div align="center">PSG analysis</div>|<div align="center">PSG analysis</div>|<div align="center">PSG analysis</div>|<div align="center">ECG analysis</div>|<div align="center">ECG analysis</div>|<div align="center">ECG analysis</div>|
