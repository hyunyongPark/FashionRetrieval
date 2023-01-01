# Fashion image Retrieval - Arcface Loss

### Reference
- https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf


### Model Description 
<table>
    <thead>
        <tr>
            <td>Arcface Architecture</td>
            <td>Comparison of Loss</td>
            <td>Arcface formula</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/KDeep_Recommendation/blob/main/img/architecture.PNG"/></td>
            <td><img src="https://github.com/hyunyongPark/KDeep_Recommendation/blob/main/img/architecture2.PNG"/></td>
            <td><img src="https://github.com/hyunyongPark/KDeep_Recommendation/blob/main/img/architecture3.PNG"/></td>
        </tr>
    </tbody>
</table>



### Requirements
- python V  # python version : 3.8.13
- dgl==0.9.1
- tqdm
- torch==1.9.1
- torchvision==0.10.1
- torchaudio==0.9.1
- torchtext==0.10.1
- dask
- partd
- pandas
- fsspec==0.3.3
- scipy
- scikit-learn


### cmd running

The install cmd is:
```
conda create -n your_prjname python=3.8
conda activate your_prjname
cd {Repo Directory}
pip install -r requirements.txt
```
- your_prjname : Name of the virtual environment to create


##### Trained weight file Download 
Download the trained weight file through the link below.
This file is a trained file that learned the k-deep fashion dataset.
Ensure that the weight file is located at "model/".
- https://drive.google.com/file/d/11bt3BocyaukuP0GNVkAeM5Aue81A1kdX/view?usp=share_link

The testing cmd is: 
```

python3 evaluation.py 

```

If you want to proceed with the new learning, adjust the parameters and set the directory and proceed with the command below.

The Training cmd is:
```

python3 training_.py 

```


### Test Result
- Pinsage Reference Result Tables in Original Paper
<table>
    <thead>
        <tr>
            <td>mAP@K(=10) Score</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src=""/></td>
        </tr>
    </tbody>
</table>


- Our Pinsage Model Performance Table

|Group criteria|Dataset|mAP@K(=50)|mAP@K(=10)|HR@K(=5)|
|---|---|---|---|---|
|category+color|train(30,570)/valid(3,804)/test(3,910)|*74.5%*|*54.8%*|*38.2%*|
|category+color+print|train(139,637)/valid(17,339)/test(17,936)|**92.6%**|**74.8%**|**49.6%**|


<table>
    </thead>
    <tbody>
        <tr>
            <td><img src=""/></td>
        </tr>
    </tbody>
</table>

- Example Result for Arcface model's query items
<table>
    <thead>
        <tr>
            <td>K-Neareast Neighbors Result</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src=""/></td>
        </tr>
    </tbody>
</table>
