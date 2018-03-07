- Start
- Step 1: Create the distribution of the existing data
	* Find the IP_frequency of each IP.
	* Create frequency of IP_frequency.
	* Use the new frequency to create the critical range of the outlier.
- Step 2: Identify the critical values for strong and week outlier from Grubbs Outlier Statistics.
	* Grubb’s statistics G_max = (X_max - X_avg)/ SD_X.
	* X_avg = 1/p ∑ x_i
	* SD_X = sqrt(1/(p-1)  ∑  (X_J - X_avg) )
	* Find G_crit(0.05) and G_crit(0.025) from the G table for the givel level of alpha.
	* If G_max <= G_crit(0.05) then "Not an outlier"
	   else If G_max > G_crit(0.05) and G_max < G_crit(0.025)  then "week outlier"
	   else "strong outlier"
- Step 4:  If the IP is not an outlier, add it to the database and update the frequency distribution.
- End
