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
The Business goal is  to come up with a supervized machine learning classification model, in this particular case, logistic regression was chosen. The output is binary, meaning that a player must get 1 to have high chance to win, or 0 to have  zero chance to win.The dataset was obtained by doing web scraping in the offical website of the PGA Tour (https://www.pgatour.com/stats), which contained the statistics collected from the tournaments played since 1980. In this particular application, the statistics used were from 2011-2021, i.e., covering only 10 years span. However, the potential user can easily covered the completed span (1980-2022) is desired by using the jupyter notebook attached.
</p>

<h2>Data Understanding</h2>
<p text-align-last="start">
As mentioned before, the dataset was scraped from the offical website of the PGA Tour, covering only ten years span of turnaments (2011-2021) in this particular case. it consists of 27 columns and 4122 rows as shown on Figure 2. The target/dependent columns is "Win" which is binary (0=did not win and 1 = won). This variable is imbalanced as will be seen later, and basically indicate winning a PGA tournament.
</p>

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
<p text-align-last="start">
All the independent variables are numerical. Before cleaning the dataset, the index was reset, and the column: 'PLAYER NAME' was dropped (see Figure 4), since it won't be needed for further analysis. The null values were identified asshown on Figures 5, and dropped. The dataset was reduced to 27 columns and  3380 rows as shown on Figure 6. Duplicates was observed in just one row as shown on Figure 7.
</p>
  
</br>
</br>
<p align="center">
<img src="images/Capstone_proj_reset.jpeg" width="800px" height="150px">
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
<img src="images/Capstone_proj_info_1.jpeg" width="500px" height="500px">
<h4 align="center"> Figure 6</h4>
</p>
</div>

</br>
<p align="center">
<img src="images/cpastone_proj_dup.jpeg" width="800px" height="150px">
<h4 align="center"> Figure 7</h4>
</p>
</div>


<p text-align-last="start">
A correlation matrix was generated in order to explore how the variables correlated each other, i.e., potentially helping to reduced the number of independent variables that will be finally used. The results are shown on Figure 8, indicating that there a number of independent variables are highly positive correlated between each other. As results, the following columns were dropped: 'Ball Speed_AVG.','SG: Tee-to-Green_SG:OTT','SG: Tee-to-Green_SG:ARG','SG: Tee-to-Green_SG:APR','SG: Putting_TOTAL SG:PUTTING','Rough Tendency_%'. Also the column: 'Scoring Average_TOTAL ROUNDS' was dropped, since it won't be needed for the upcoming analysis.
</p>


</br>
<p align="center">
<img src="images/CapstoneProj_matrixcorrelation.jpeg" width="900px" height="800px">
<h4 align="center"> Figure 8</h4>
</p>
</div>

<p text-align-last="start">
More insight into the dataset can be gained before finalizing the data preparation by showing the values distribution for all the independent variables as shown on Figure 9. As it can be observed, all of them show a quite nice normal distribution. Also, it is evident that there are no outliers present.
</p>

</br>
<p align="center">
<img src="images/CaptsoneProj_distindepvaria_1.jpeg" width="900px" height="800px">
<h4 align="center"> Figure 9</h4>
</p>
</div>

<p text-align-last="start">
The target column, i.e., the dependent variables: "Win" is binary (0=did not win and 1= won), and imbalanced as shown on Figure 10:
</p>

</br>
</br>
<p align="center">
<img src="images/CapstoneProj_dependvar.jpeg" width="800px" height="600px">
<h4 align="center"> Figure 10</h4>
</p>

<p text-align-last="start">
A good understanding on the differences in statistics among the players who won and the ones who did not throughout the years (2011-2021) can be obtained using boxplots: independent variables vs. years, using the dependent variable: 'Win' as a hue as seen on Figure 11. It can be seen that the players who won performed much better in all the categories/statistics, in particular the follow ones:  'Scoring Average_AVG', 'Driving Distance_AVG', '  Club Head Speed_AVG.', 'SG: Off-the-Tee_TOTAL SG:OTT', ' SG: Approach the Green_TOTAL SG:APP', and 'SG: Tee-to-Green_AVERAGE'. Also note that the difference in 'SG: Putting_AVERAGE' is not significant  between players regardless if they won or did not, indicating that putting is not the most important factor to win or lose a tournament in the PGA tour. Another important observation is that the only two category/statistics that have improved through the years are: 'Driving Distance_AVG', and '  Club Head Speed_AVG.'. The major reasons behind it are : golf players are more athletic than ever, and equipment technology has dramatically improved through the years, specially the last years.
</p>

</br>
<p align="center">
<img src="images/capstone_proj_box_1.jpeg" width="1200px" height="800px">
<img src="images/capstone_proj_box_2.jpeg" width="1200px" height="800px">
<img src="images/capstone_proj_box_3.jpeg" width="12000px" height="400px">
<h4 align="center"> Figure 11</h4>
</p>
</div>

<h3>Splitting the variables</h3>
<p text-align-last="start">
Splitting the dependent variable from the independent variables and assigning them to y and X respectively was done as shown on Figure 12:
</p>

</br>
<p align="center">
<img src="images/capstone_proj_split.jpeg" width="1000px">
<h4 align="center"> Figure 12</h4>
</p>


<h3>Cross-Validation Approach</h3>
<p text-align-last="start">
Although, the dependent variable is imbalanced, the HoldOut Cross-validation was used. In this technique, the whole dataset is randomly partitioned into a training set and validation set using the train_test_split function. The stratify parameter was used  to preserve  the dataset proportions for better prediction and reproduceability of results as shown on Figure 13:
</p>

</br>
</br>
<p align="center">
<img src="images/capstone_proj_cross.jpeg" width="1000px">
<h4 align="center"> Figure 13</h4>
</p>

<h2>Modelling</h2>
<p text-align-last="start">
It is well known that working with imbalance data is always a challenge for any particular Machine Learning Classification Model. In this particular piece of work, Logistic Regression model was chosen as the Supervise Machine Learning Model due to the binary nature of the dependent/target variable (has only 2 outputs), and its capacity to predict the likelihood of events by looking at historical data points. However, as any machine learning algorithm, Logistic Rgression works much better when its hyperparamters are optimized using the appropriate metric. In this particular project, the metric chosen was: 'roc_auc' (the area under the ROC curve), since it works quite well for imbalance data. The metric called 'Recall' was tested as scoring to optimize the hyperparamters, but results were not encouraged. The pipeline model used is shown on Figure 14. 
</p>

</br>
<p align="center">
<img src="images/capstone_proj_pipe.jpeg" width="250px">
<h4 align="center"> Figure 14</h4>
</p>

The GridSearchCV function was used to optimized several hyper-parameters tested (see Figure 15). The optimum values for those, and the metrics results are shown on Figure 16. 

</br>
<p align="center">
<img src="images/" width="1000px">
<h4 align="center"> Figure 15</h4>
</p>

</br>
<p align="center">
<img src="images/" width="1000px">
<h4 align="center"> Figure 16</h4>
</p>




