# ADVSCD: Anomaly Driven Video Summarization Using Change Detection

## Objective
Perform video summarization on lengthy CCTV footage focused on public crimes.

## Dataset Used:

**UCF-Crime Dataset** : https://www.crcv.ucf.edu/projects/real-world/

## Highlights

- Efficiently summarizes large videos around crime anomalies present in the UCF-crime dataset when tested on a small-scale
- Introduces a novel quantitative evaluation of video summarization
- Compactness of the output video

The figure below shows the algorithm's effectiveness when tested on a few randomly chosen videos from the UCF-crime dataset.

<img width="700" alt="image" src="https://github.com/bipinthecoder/anomaly-driven-video-summarization/assets/37789083/9522578e-94d1-4df1-bf2e-4bfd0c700e46">


## Proposed Architecture

The final architecture followed in the project is shown below:

<img width="700" alt="image" src="https://github.com/bipinthecoder/anomaly-driven-video-summarization/assets/37789083/8ac6a2b4-52fd-41a5-bc54-682a077e2d97">


## Stages of the Architecture

### 1. CDM (Change Detection Module)

This module identifies the frames undergoing significant changes in the scene and passes that to the next stage of the architecture.

The figure below illustrates the operations happening in the CDM module:

<img width="700" alt="image" src="https://github.com/bipinthecoder/anomaly-driven-video-summarization/assets/37789083/10ccedd1-f216-474c-88bf-3e2ffc60a7d1">


### 2. ADM (Anomaly Detection Module)

This module identifies the crime anomalies happening in the scenes and passes the potential anomalies to the next stage of the architecture.

The figure below shows the architecture of the adopted LSTM-based pre-trained Autoencoder fine-tuned on the UCF-crime dataset:

<img width="700" alt="image" src="https://github.com/bipinthecoder/anomaly-driven-video-summarization/assets/37789083/5680e64c-ce40-4a09-8ff5-eb0650295aca">


### 3. CBT (Clustering-Based Technique)

K-means clustering is performed on the selected anomalies for categorizing critical anomalies without any manual intervention.

<img width="700" alt="image" src="https://github.com/bipinthecoder/anomaly-driven-video-summarization/assets/37789083/e6b43358-63fb-4d88-98a7-8f5105438d6b">



### 4. VSM (Video Summarization Module)

This module performs video summarization on selected frames by combining some of the frames from the original video to generate a meaningful compact video output centred around the anomaly.




## State-of-the-art adopted either as it is or modified for this project
<table style="width:100%;">
    <tr>
        <th>Literature</th>
        <th>Aim</th>
        <th>Adopted Idea</th>
        <th>Part of the proposed Architecture</th>
    </tr>
    <tr>
        <td><a href="https://www.researchgate.net/publication/287454605_Mov- ing_Object_Detection_and_Segmentation_using_Frame_Differencing_and_Summing_ Technique">Thapa et al.</a></td>
        <td>Moding Object Detection</td>
        <td>Frame Differencing/Summing</td>
        <td>Change Detection</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1701.01546">Chong and Tay</a></td>
        <td>Anomaly Detection in Videos</td>
        <td>Spatiotemporal Autoencoder, ConvLSTM</td>
        <td>Anomaly Detection Model</td>
    </tr>
   <tr>
        <td><a href="https://arxiv.org/abs/1604.04574">Hasan et al.</a></td>
        <td>Learning Regularity in Videos</td>
        <td>Use of Autoencoder, Reconstruction Score</td>
        <td>Anomaly Detection</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1910.04792">Jadon and Jasim</a></td>
        <td>Video Summarization</td>
        <td>Video Skimming</td>
        <td>Video Summarization</td>
    </tr>
</table>

## Evaluation
This project evaluates the effectiveness of the final architecture by considering the SSIM (Structural Similarity Index Method), IP (Inclusion Percentile of Anomalies), Compactness Measure and G-mean value. A multi-objective evaluation technique like Pareto Front is also utilized for evaluating the architecture. This has not been incorporated into the repository but will be clearly explained in a paper published in future.

## How to Run the project

- Clone the repository
- Create a Virtual Env using conda from the environment.yml file: `conda env create -f environment.yml`
- Set up the config.py file by providing the right path to relevant files. The suggested value of NUMBER_OF_CLUSTERS is 3 for the UCF-crime dataset.
- The execution entry point is main_cluster_based.py, and hence the command `python main_cluster_based.py` will perform video summarization.
- The output video will be in the directory specified as `TO_SAVE_DIRECTORY` in the `config.py` file.

## File Structure
### Models
All the models associated with this project are in the directory `/models`. The final fine-tuned model selected for the architecture after several tests is `auto_encoder5.hdf5`.

### Notebooks
All the associated Notebooks are in the directory `/notebooks`. All the model training, visualization and unit testing were performed in the respective notebooks. The notebooks, however, are not properly cleaned and hence have minimal readability.

### Utils
All the separate module functions and repetitively used helper functions are written in respective files and stored in the `/utils` directory.

## Remarks and Open Issues

### Initial Approach and its future scope
- The file `main.py` contains an approach that utilizes potentially non-anomalous frames to create a threshold value for frames flagged as anomalous.
- In the future, this method could be developed by adding feedback loops with a scope for improvement.
### Limitations
- This architecture does not do well on subtle human crime anomalies like Pickpocketing, Shoplifting, etc
- This can be overcome by training the model on more videos and improvising on this architecture.
  
## Acknowledgements
- The pre-trained LSTM Autoencoder trained on <a href="http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm">USCD Pedestrian Dataset</a> was obtained from the GitHub public repo : https://github.com/hashemsellat/video-anomaly-detection/tree/master
- The script for plotting the Pareto front was obtained from the GitHub public repo: https://github.com/Mohamed-Zeghlache/Pareto-frontier


