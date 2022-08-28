<div>
<img src="images/Capstone_proj_logo.jpeg" alt="isolated"  width="250px" align="left">
<h1> PREDICTING PGA TOUR PLAYER'S CHANCE TO WIN A PGA TOUR-TOURNAMENT</h1>
<p text-align= "right">
This python application using jupyter notebook predicts the chance of any pga tour player to win a tournament using Logistic Regression as a Supervized Machine Learning Algorithm.
</p>
</div>

</br>
</br>
</br>
<div>
  <p>The current CRISP-DM Process Model for Data Mining (see Figure 1) was followed.</p>

</br>
</br>
<p align="center">
<img src="images/Figure1_CRISP_DM_Model.jpeg" width="300px" height="300px">
<h4 align="center"> Figure 1</h4>
</p>
</div>

<h2>Business Understanding</h2>
<p text-align-last="start">
The Business goal is  to come up with a supervized machine learning classification model, in this particular case, logistic regression was chosen. The output is bynary, meaning that a player must get 1 to have high chance to win, or 0 to have  zero chance to win.The dataset was obtained by doing web scraping in the offical website of the PGA Tour (https://www.pgatour.com/stats), which contained the statistics collected from the tournaments played since 1980. In this particular application, the statistics used were from 2011-2021, i.e., covering only 10 years span. However, the potential user can easily covered the completed span (1980-2022) is desired by using the jupyter notebook attached.
</p>

<h2>Data Understanding</h2>
As mentioned before, the dataset was scraped from the offical website of the PGA Tour, covering only ten years span of turnaments (2011-2021) in this particular case. it consists of 27 columns and 4122 rows as shown on Figure 2. The target/independent columns is "Win" which is binary (0 and 1). This variable is imbalanced as will be seen later. 

</br>
</br>
<p align="center">
<img src="images/CapstoneProj_info_2.jpeg" width="500px" height="500px">
<h4 align="center"> Figure 2</h4>
</p>
</div>

</br>
</br>
<p align="center">
<img src="images/Capstone_proj_head-1.jpeg" width="900px" height="500px">
<h4 align="center"> Figure 3</h4>
</p>
</div>


<h2>Data Preparation</h2>
All the independent variables are numerical. Before cleaning the dataset, the index was reset, and the column: 'PLAYER NAME' was dropped (see Figure 4), since it won't be needed for further analysis. The null values were identified shown on Figures 5, and dropped. The dataset was reduced to 27 columns and  3380 rows as shown on Figure 6. Duplicates were not observed.

</br>
</br>
<p align="center">
<img src="images/Capstone_proj_reset.jpeg" width="800px" height="200px">
<h4 align="center"> Figure 4</h4>
</p>
</div>

</br>
<p align="center">
<img src="images/Capstone_proj_nulls.jpeg" width="900px"  height="500px">
<h4 align="center"> Figure 5</h4>
</p>
</div>


</br>
<p align="center">
<img src="images/Capstone_proj_info_1.jpeg" width="600px" height="500px">
<h4 align="center"> Figure 6</h4>
</p>
</div>

