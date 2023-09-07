import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

gender_options = ['F', 'M']
region_category_options = ['City', 'Village', 'Town']
membership_category_options = ['No Membership', 'Basic Membership', 'Silver Membership',
                               'Premium Membership', 'Gold Membership', 'Platinum Membership']
joined_through_referral_options = ['Yes', 'No']
preferred_offer_types_options = ['Credit/Debit Card Offers', 'Gift Vouchers/Coupons','Without Offers']
medium_of_operation_options = ['Desktop', 'Smartphone', 'Both']
internet_option_options = ['Wi-Fi', 'Fiber_Optic', 'Mobile_Data']
used_special_discount_options = ['Yes', 'No']
offer_application_preference_options = ['Yes', 'No']
past_complaint_options = ['Yes', 'No']
complaint_status_options = ['No Information Available', 'Not Applicable', 'Unsolved', 'Solved',
                            'Solved in Follow-up']
feedback_options = ['Poor Website', 'Poor Customer Service', 'Too many ads', 'Poor Product Quality',
                    'No reason specified', 'Products always in Stock', 'Reasonable Price',
                    'Quality Customer Care', 'User Friendly Website']

def run():
    with open('final_pipeline.pkl', 'rb') as file_1:
        model_pipeline = pickle.load(file_1)
        model_ann = load_model('churn_model.h5')

    with st.form('key=form_customer'):
        st.write('Data Customer')
        name = st.text_input('Nama Lengkap', value='')
        user_id = st.text_input('ID pengguna', value='')
        age = st.number_input('Umur', min_value=1, max_value=100,value=35,step=1,help='Usia Customer')
        gender = st.selectbox("Gender", gender_options)
        st.write("---")
        region_category = st.selectbox("Region Category", region_category_options)
        membership_category = st.selectbox("Membership Category", membership_category_options)
        joining_date = st.date_input("Joining Date")
        joined_through_referral = st.selectbox("Joined Through Referral", joined_through_referral_options)
        st.write("---")
        preferred_offer_types = st.selectbox("Preferred Offer Types", preferred_offer_types_options)
        offer_application_preference = st.selectbox("Offer Application Preference", offer_application_preference_options)
        st.write("---")
        medium_of_operation = st.selectbox("Medium of Operation", medium_of_operation_options)
        internet_option = st.selectbox("Internet Option", internet_option_options)
        st.write("---")
        last_visit_time = st.number_input("Last Visit Time",min_value=1, max_value=5000,value=35,step=1)
        days_since_last_login = st.number_input("Days Since Last Login", min_value=0, max_value=30,value=10,step=1)
        avg_time_spent = st.number_input("Average Time Spent", min_value=1, max_value=5000,value=35,step=1)
        avg_frequency_login_days = st.number_input("Average Frequency Login Days", min_value=1, max_value=100,value=30,step=1)
        st.write('---')
        avg_transaction_value = st.number_input("Average Transaction Value", min_value=0, max_value=100000,value=35000,step=1)
        points_in_wallet = st.number_input("Points in Wallet", min_value=1, max_value=50000,value=750,step=1)
        used_special_discount = st.selectbox("Used Special Discount", used_special_discount_options)
        st.write("---")
        past_complaint = st.selectbox("Past Complaint", past_complaint_options)
        complaint_status = st.selectbox("Complaint Status", complaint_status_options)
        feedback = st.selectbox("Feedback", feedback_options)

        submit_button = st.form_submit_button('Determine')

    data_dict = {
    'user_id': user_id,
    'age': age,
    'gender': gender,
    'region_category': region_category,
    'membership_category': membership_category,
    'joining_date': str(joining_date),
    'joined_through_referral': joined_through_referral,
    'preferred_offer_types': preferred_offer_types,
    'medium_of_operation': medium_of_operation,
    'internet_option': internet_option,
    'last_visit_time': str(last_visit_time),
    'days_since_last_login': days_since_last_login,
    'avg_time_spent': avg_time_spent,
    'avg_transaction_value': avg_transaction_value,
    'avg_frequency_login_days': avg_frequency_login_days,
    'points_in_wallet': points_in_wallet,
    'used_special_discount': used_special_discount,
    'offer_application_preference': offer_application_preference,
    'past_complaint': past_complaint,
    'complaint_status': complaint_status,
    'feedback': feedback}

    st.write('---')
    st.write('#### Informasi ',name)
    data_inf = pd.DataFrame([data_dict])
    st.dataframe(data_inf)

    data_inf_transform = model_pipeline.transform(data_inf)
    y_pred_inf = model_ann.predict(data_inf_transform)
    y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)
    st.write('Churn Status: ',y_pred_inf[0][0])
    st.write('#### Aksi:')
    if y_pred_inf[0][0]==0:
        st.write('Customer tidak perlu tindakan')
    else:
        if offer_application_preference == "Yes": 
            st.write(f'Segera berikan {preferred_offer_types} sebelum {avg_frequency_login_days} hari via {medium_of_operation}')
        else:
            st.write('komplain belum bisa kami penuhi sekarang, terima kasih atas masukkan nya')

    if y_pred_inf[0][0]==1:
        if complaint_status == 'Unsolved':
                st.write(f'Segera konfirmasi komplain {feedback} akan kita diselesaikan')
        elif complaint_status == 'Solved' or complaint_status =='Solved in Follow-up':
            st.write(f'Segera konfirmasi komplain {feedback} sudah diselesaikan, silahkan menggunakan jasa kami kembali')
        else:
            st.write('komplain belum bisa kami penuhi sekarang, terima kasih atas masukkan nya, kami berharap anda tetap menggunakan jasa kami')

if __name__ == '__main__':
    run()


