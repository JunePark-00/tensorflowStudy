**결론부터 말하자면 batch size와 성능의 상관 관계는 아직 정확하게 규정되지는 않았습니다.**

***task, 데이터에 따라 그 기준이 달라지기 때문***입니다.

***다만, 일반적으로 32, 64 크기의 mini-batch가 성능에는 가장 좋다고 알려져 있습니다.***

**batch size를 줄이거나 늘임으로써 얻는 장점을 요약하자면 다음과 같습니다.**

****

**▶ batch size를 줄임으로써 얻는 장점**

**- 필요한 메모리 감소: 전체 데이터를 쪼개어 여러 번 학습하는 것이기 때문에 최소 요구 메모리량을 줄일 수 있음.**

****

**▶ batch size를 늘임으로써 얻는 장점**

**- 아래 graph를 보면 전체 데이터를 활용한 Batch의 경우(파란색 그래프)보다 batch size가 작은 Mni-batch의 경우(초록색 그래프)가 더 fluctuate 한 것을 확인할 수 있음.**

**(더 flucatuate 하다는 것은 학습이 불안정 해진다는 의미)**

****

![img](https://mblogthumb-phinf.pstatic.net/MjAxOTAxMjRfMjc2/MDAxNTQ4MjYwMTAzMDAy.4v8IAItG1q1T82S4df42xakQk7xN9mSZezTvk2Sf54Eg.N9Oc4vhdfO7AJBWRgfn2Iufvnv-c3HPAbeHwuF_Ypokg.PNG.qbxlvnf11/lU3sx.png?type=w800)

****

**▶ 정리**

**가용 메모리가 적을 때는 batch size를 상대적으로 작게,**

**보다 안정적으로 학습을 시키고 싶다면 batch size를 상대적으로 높게 설정해주면 됩니다.**

**다만, ** ***batch size가 커질 수록 일반화 성능은 감소하는 경우가 다소 확인이 되었으니 그 점만 유의해주시면 되겠습니다.***

