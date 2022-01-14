# cg2021f-hw12
## 安裝環境
* 創建虛擬環境
```
conda create -n cg2021f-hw12 python=3.7
```
* 啟動虛擬環境
```
conda activate cg2021f-hw12
```
* 安裝 pytorch
```
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```
* 安裝其他套件
```
pip install -r requirements.txt
```
## 影像分割
```
python predict.py
```
## 輸出nrrd檔
```
python write_nrrd.py
```
