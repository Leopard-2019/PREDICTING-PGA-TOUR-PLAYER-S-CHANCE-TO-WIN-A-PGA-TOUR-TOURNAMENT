<div>
<img src="images/Capstone_proj_logo.jpeg" alt="isolated"  width="250px" align="left">
<h1> PREDICTING PGA TOUR PLAYER'S CHANCE TO WIN A PGA TOUR-TOURNAMENT</h1>
<p text-align= "right">
This python application using jupyter notebook predicts the chance of any pga tour player to win a tournamet using Logistic Regression as a supervized machine learning algorithm.
</p>
</div>

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
As mentioned before, the dataset was scraped from the offical webiste of the PGA Tour, covering only ten years span of turnament (2011-2021) in this particular case. it consists of 27 columns and 4122 rows as shown below. The target/independent columns is "Win" which is binary. This variable is imbalance as will be later be seen. All the independent variables are numerical. After dropping the "NaN" values, the dataset was reduced to 27 columns and  3380 rows. Duplicates were not observed.
