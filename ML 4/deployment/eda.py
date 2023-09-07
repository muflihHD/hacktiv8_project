import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway


df = pd.read_csv('churn.csv')
col_df = list(df.columns)
df_num_columns = df.select_dtypes(include=np.number).columns.tolist()
df_cat_columns = df.select_dtypes(include=['object']).columns.tolist()
cat_col_filtered = ['gender', 'region_category', 'membership_category', 'joined_through_referral', 'preferred_offer_types', 'medium_of_operation', 'internet_option', 'used_special_discount', 'offer_application_preference', 'past_complaint', 'complaint_status', 'feedback']

def normal_bound(variable,printout=False,data=df):
    upper_boundary = round((data[variable].mean() + 3 * data[variable].std()),2)
    lower_boundary = round((data[variable].mean() - 3 * data[variable].std()),2)
    if printout:
      st.write(f'right end outliers (> ',upper_boundary,'):',len(data[data[variable]  > upper_boundary]))
      st.write(f'left end outliers  (< ',lower_boundary,'):', len(data[data[variable]  < lower_boundary]))
      st.write('% right end outliers:',round(len(data[data[variable] > upper_boundary]) / len(data[variable]) * 100, 2),'%' )
      st.write('% left end outliers :',round(len(data[data[variable] < lower_boundary]) / len(data[variable]) * 100, 2),'%' )
    total_outlier = (len(data[data[variable]  > upper_boundary]))+(len(data[data[variable]  < lower_boundary]))
    return upper_boundary, lower_boundary, total_outlier

def skewed_bound(variable,printout=False, distance=1.5,data=df):
    IQR = data[variable].quantile(0.75) - data[variable].quantile(0.25)
    lower_boundary = round((data[variable].quantile(0.25) - (IQR * distance)),2)
    upper_boundary = round((data[variable].quantile(0.75) + (IQR * distance)).round(2),2)
    if printout:
      st.write(f'right end outliers (> ',upper_boundary,'):',len(data[data[variable]  > upper_boundary]))
      st.write(f'left end outliers  (< ',lower_boundary,'):', len(data[data[variable]  < lower_boundary]))
      st.write('% right end outliers:',round(len(data[data[variable] > upper_boundary]) / len(data[variable]) * 100, 2),'%' )
      st.write('% left end outliers :',round(len(data[data[variable] < lower_boundary]) / len(data[variable]) * 100, 2),'%' )
    total_outlier = (len(data[data[variable]  > upper_boundary]))+(len(data[data[variable]  < lower_boundary]))
    return upper_boundary, lower_boundary, total_outlier

def run():
    st.header('Customer Churn Dataset')
    st.subheader('Muflih Hafidz Danurhadi RMT 020')

    st.markdown('---')
    st.write('Untuk Analisa awal dengan visualisasi')

    st.dataframe(df)

    st.write('#### Visualisasi')
    opsi = st.selectbox('pilih kolom:',(col_df[1:]))
    if opsi in df_num_columns:
      st.write(f'#### Distribusi {opsi}')
      fig = plt.figure(figsize=(15,10))
      plt.axvline(df[opsi].mean(), color='magenta', linestyle='dashed', linewidth=2)
      plt.axvline(df[opsi].median(), color='green', linestyle='dashed', linewidth=2)
      plt.axvline(df[opsi].mode()[0], color='blue', linestyle='dashed', linewidth=2)
      sns.histplot(df[opsi],bins=30,kde=True)
      st.pyplot(fig)
      st.write(f'Mean {opsi}  :',round(df[opsi].mean(),2))
      st.write(f'Median {opsi}:',df[opsi].median())
      st.write(f'Mode {opsi}  :',df[opsi].mode()[0])
      st.write(f'Max {opsi}   :',df[opsi].max())
      st.write(f'Min {opsi}   :',df[opsi].min())
      st.write(f'Std {opsi}   :',round(df[opsi].std(),2))
      st.write('---')
      st.write(f'#### Outlier {opsi}')
      skew = df[opsi].skew().round(2)
      fig2, ax = plt.subplots(figsize=(10, 10))
      sns.boxplot(y=df[opsi])
      plt.title(f'Boxplot {opsi}')
      st.pyplot(fig2)
      st.write('Skewness Value:', skew)
      if  -0.5<= skew <=0.5:
          st.write(f'Distribusi {opsi} Normal')
          normal_bound(opsi,True)
      else:
          st.write(f'Distribusi {opsi} Skewed')
          skewed_bound(opsi,True)
      st.write('---')
      st.write(f'#### Anova test { opsi}')
      group_positive = df[df['churn_risk_score'] == 1][opsi]
      group_negative = df[df['churn_risk_score'] == 0][opsi]
      statistic, p_value = f_oneway(group_positive, group_negative)

      st.write(f"ANOVA Results {opsi}:")
      st.write("F-statistic:", round(statistic,3))
      st.write("p-value:", round(p_value,3))

      if p_value < 0.05 and opsi != 'churn_risk_score' :
          st.write(f"Ada hubungan yang signifikan antara {opsi} dan churn_risk_score")
      else:
          st.write(f"Tidak ada hubungan yang signifikan antara {opsi} dan churn_risk_score")
    elif opsi in cat_col_filtered:
      st.write(f'#### Jumlah {opsi}')
      fig3, ax = plt.subplots(figsize=(10, 10))
      bars = df[opsi].value_counts()
      plt.barh(bars.index, bars.values, color=sns.palettes.mpl_palette('Dark2'))
      plt.title(opsi)
      ax.spines[['top', 'right',]].set_visible(False)
      st.pyplot(fig3)
      st.write(f'{opsi} Values Percentage')
      jumlah = df[opsi].value_counts()
      total = len(df)
      percentages = round((jumlah / total) * 100,2)
      st.write(percentages)
      st.write('---')
      st.write(f'#### Jumlah {opsi} terhadap Churn')
      fig4, ax = plt.subplots(figsize=(10, 10))
      df_2dhist = pd.DataFrame({
          x_label: grp[opsi].value_counts()
          for x_label, grp in df.groupby('churn_risk_score')
      })
      sns.heatmap(df_2dhist, cmap=sns.cubehelix_palette(start=.5, rot=-.8))
      plt.xlabel('churn_risk_score')
      plt.ylabel(opsi)
      st.pyplot(fig4)
      
      highest_death = df.groupby(opsi)['churn_risk_score'].sum().idxmax()
      lowest_death = df.groupby(opsi)['churn_risk_score'].sum().idxmin()
      highest_survived = df[df['churn_risk_score'] == 0].groupby(opsi)['churn_risk_score'].count().idxmax()
      lowest_survived = df[df['churn_risk_score'] == 0].groupby(opsi)['churn_risk_score'].count().idxmin()

      count_death_highest = df.groupby(opsi)['churn_risk_score'].sum().max()
      count_death_lowest = df.groupby(opsi)['churn_risk_score'].sum().min()
      count_survived_highest = df[df['churn_risk_score'] == 0].groupby(opsi)['churn_risk_score'].count().max()
      count_survived_lowest = df[df['churn_risk_score'] == 0].groupby(opsi)['churn_risk_score'].count().min()

      st.write(f"kelompok {opsi} {highest_death} memiliki churn (1) tertinggi sejumlah :",count_death_highest)
      st.write(f"kelompok {opsi} {highest_survived} memiliki not churn (0) tertinggi sejumlah :",count_survived_highest)
      st.write(f"kelompok {opsi} {lowest_death} memiliki churn terendah (1) sejumlah :", count_death_lowest)
      st.write(f"kelompok {opsi} {lowest_survived} memiliki not churn terendah (0) sejumlah :", count_survived_lowest)
      st.write('---')
      st.write(f'#### Uji Chi Square { opsi}')
      cross_tab = pd.crosstab(df[opsi], df['churn_risk_score'])
      chi2, p_value, dof, expected = chi2_contingency(cross_tab)
      st.write(f"Hasil Uji Chi-Square {opsi}:")
      st.write("Chi-Square:", round(chi2,3))
      st.write("Nilai p-value:", round(p_value,3))

      if p_value < 0.05 and opsi != 'churn_risk_score':
          st.write(f"Ada hubungan yang signifikan antara {opsi} dan churn_risk_score")
      else:
          st.write(f"Tidak ada hubungan yang signifikan antara {opsi} dan churn_risk_score")
    else:
       st.write('Grafik untuk data tersebut belum dibuat')

if __name__ == '__main__':
    run()
