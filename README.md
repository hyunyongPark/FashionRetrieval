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
            <td><img src="https://github.com/hyunyongPark/FashionRetrieval/blob/main/img/architecture.PNG"/></td>
            <td><img src="https://github.com/hyunyongPark/FashionRetrieval/blob/main/img/comparison.PNG"/></td>
            <td><img src="https://github.com/hyunyongPark/FashionRetrieval/blob/main/img/formula.PNG"/></td>
        </tr>
    </tbody>
</table>



### Requirements
- python V  # python version : 3.8.13
- timm
- tqdm
- torch==1.9.1
- torchvision==0.10.1
- torchaudio==0.9.1
- torchtext==0.10.1
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

 
To view the test results, we first embed the entire DB images, and then learn and store the ANN algorithm.
Afterwards, the stored annoy (ANN) predicts the pseudo group of the test set and saves it as a pkl
Finally, the mAP is obtained by obtaining an intersection between the predicted group and the correct answer.

The testing cmd is: 
```
python3 generate_DBembedding.py
python3 eval_savingANN.py
eval_testsetPrediction.py
python3 evaluation_performance.py 

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
            <td><img src="https://github.com/hyunyongPark/FashionRetrieval/blob/main/img/test_result.PNG"/></td>
        </tr>
    </tbody>
</table>


- Our Pinsage Model Performance Table

|Group criteria|Dataset|mAP@K(=50)|mAP@K(=10)|
|---|---|---|---|
|category+color|train(139,637)/valid(17,339)/test(17,936)|**82.6%**|**47.28%**|


<table>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/FashionRetrieval/blob/main/img/test_result.PNG"/></td>
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
