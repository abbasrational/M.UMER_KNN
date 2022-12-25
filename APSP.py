import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
fig = go.Figure()
import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import *
from PIL import Image
from sklearn.pipeline import Pipeline
import pickle 

st.set_page_config(page_title="An Intelligent TL Faults Analysis",page_icon="🧊")
################################################################################### M A I N    P A G E ############################
Side=st.sidebar.selectbox("PROJECT DETAILS: ",('Main Page','Data Visualizations','Prediction Results', 'Real Time Predictions'))
if Side=="Main Page":
    #st.markdown("### Select Demo")
    st.write('')
    st.write("# An Intelligent Web Dashboard for Transmission Line Faults Analysis")
    #st.markdown("# Precious Metals Price Predictions")
    #st.sidebar.header("# Metals Prediction Plots")
    st.markdown(
        """
        PROJECT DETAILS 
        **👈 Select a demo from the sidebar** to visualize the Results!
        ### Schematic Working Visual:
    """
    )
    image = Image.open('Circuit.png')
    st.image(image, caption='')
################################################################# 2nd selection ######################################################       ################################################################# 2nd selection ######################################################  
################################################################# 2nd selection ######################################################  
################################################################# 2nd selection ######################################################  
################################################################# 2nd selection ######################################################

if Side=="Data Visualizations":

    fig = go.Figure()
#https://plotly.com/python/subplots/
    #st.sidebar.subheader("Choose Fault Resistance")

    option = st.sidebar.radio('Choose Fault Resistance',['0.1 Ohm','0.001 Ohm'])
    #############################################################################
    if option=="0.1 Ohm":

        AB_0_1=pd.read_csv("AB_0.1.csv")
        ABC_0_1=pd.read_csv("ABC_0.1.csv")
        ABG_0_1=pd.read_csv("ABG_0.1.csv")
        AC_0_1=pd.read_csv("AC_0.1.csv")
        ACG_0_1=pd.read_csv("ACG_0.1.csv")
        AG_0_1=pd.read_csv("AG_0.1.csv")
        BC_0_1=pd.read_csv("BC_0.1.csv")
        BCG_0_1=pd.read_csv("BCG_0.1.csv")
        BG_0_1=pd.read_csv("BG_0.1.csv")
        CG_0_1=pd.read_csv("CG_0.1.csv")
        No_0_1=pd.read_csv("No_Fault_0.1.csv")

        x = st.sidebar.selectbox('Choose No Fault or Fault Type',('No Fault','AB','ABC','ABG','AC','ACG','AG','BC','BCG','BG','CG'))

        if x=='AB':
            f=AB_0_1
            st.subheader("AB Fault Analysis:")
        elif x=='ABC':
            f=ABC_0_1
            st.subheader("ABC Fault Analysis:")
        elif x=='ABG':
            f=ABG_0_1
            st.subheader("ABG Fault Analysis:")
        elif x=='AC':
            f=AC_0_1
            st.subheader("AC Fault Analysis:")
        elif x=='ACG':
            f=ACG_0_1
            st.subheader("ACG Fault Analysis:")
        elif x=='AG':
            f=AG_0_1
            st.subheader("AG Fault Analysis:")
        elif x=='BC':
            f=BC_0_1
            st.subheader("BC Fault Analysis:")
        elif x=='BCG':
            f=BCG_0_1
            st.subheader("BCG Fault Analysis:")
        elif x=='BG':
            f=BG_0_1
            st.subheader("BG Fault Analysis:")
        elif x=='CG':
            f=CG_0_1
            st.subheader("CG Fault Analysis:")
        elif x=='No Fault':
            f=No_0_1
            st.subheader("No Fault Condition:")

        fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Va'],name='Va'))
        fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Vb'],name='Vb'))
        fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Vc'],name='Vc'))
        fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Ia'],name='Ia'))
        fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Ib'],name='Ib'))
        fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Ic'],name='Ic'))
        fig.update_layout(title_text="Combined Data Visualizations (Rf = 0.1 Ohms, TL Length = 300 Km)")
        fig.update_layout(xaxis_title='3-Phase Voltages (Volts) & Currents (Amp)')
        fig.update_layout(yaxis_title='Time')
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig,use_container_width=True)
    #######################################################################################################################################    
        option2 = st.sidebar.radio('Choose Phases (A, B or C)',['Phase: A','Phase: B','Phase: C'])
            
        if option2 == 'Phase: A':
    #         fig = make_subplots(rows=2, cols=1)
    #         fig.append_trace(go.Scatter(x=f['Domain'],y=f[' Va'],name='Va (Volts)'), row=1, col=1)
    #         fig.update_layout(title_text="Phase A Data Visualizations (Rf = 0.1 Ohms, TL Length = 300 Km)")
    #         fig.append_trace(go.Scatter(x=f['Domain'],y=f[' Ia'],name='Ia (Amps)'), row=2, col=1)
    #         fig.update_layout(yaxis_title='Time')
    #         st.plotly_chart(fig,use_container_width=True) 

            # Initialize figure with subplots
            fig = make_subplots(
                rows=2, cols=1 
                #,subplot_titles=("Va vs Time", "Ia Vs Time")
            )

            # Add traces
            if st.sidebar.checkbox("Show Transient Analysis", False):
                st.subheader("Approximate Transient Analysis:")
                fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Ia'],name='Ia'), row=1, col=1)
                fig.add_vline(x=0.501, line_width=3, line_dash="dash", line_color="red")
                fig.add_annotation(x=0.57, y=1.989, text="Subtransient", showarrow=False, arrowhead=1)
                fig.add_vline(x=0.64, line_width=3, line_dash="dash", line_color="black")
                fig.add_annotation(x=0.75, y=1.989, text="Transient", showarrow=False, arrowhead=1)
                fig.add_vline(x=0.85675, line_width=3, line_dash="dash", line_color="green")
                fig.add_annotation(x=0.928, y=1.989, text="Steady State", showarrow=False, arrowhead=1)
            else:
                fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Ia'],name='Ia'), row=1, col=1)
                fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Va'],name='Va'), row=2, col=1)
                
                
            # Update xaxis properties
            fig.update_xaxes(title_text="Amp", row=1, col=1)
            fig.update_xaxes(title_text="Volts", row=2, col=1)


            # Update yaxis properties
            fig.update_yaxes(title_text="Time", row=1, col=1)
            fig.update_yaxes(title_text="Time", row=2, col=1)


            # Update title and height
            fig.update_layout(title_text="Phase A Data Visualizations (Rf = 0.1 Ohms, TL Length = 300 Km)", height=700)
            st.plotly_chart(fig,use_container_width=True)


        if option2 == 'Phase: B': 
    #         fig = make_subplots(rows=2, cols=1)
    #         fig.append_trace(go.Scatter(x=f['Domain'],y=f[' Vb'],name='Vb (Volts)'), row=1, col=1)
    #         fig.update_layout(title_text="Phase B Data Visualizations (Rf = 0.1 Ohms, TL Length = 300 Km)")
    #         fig.append_trace(go.Scatter(x=f['Domain'],y=f[' Ib'],name='Ib (Amps)'), row=2, col=1)
    #         fig.update_layout(yaxis_title='Time')
    #         st.plotly_chart(fig,use_container_width=True)

            # Initialize figure with subplots
            fig = make_subplots(
                rows=2, cols=1 
                #,subplot_titles=("Va vs Time", "Ia Vs Time")
            )

                        # Add traces
            if st.sidebar.checkbox("Show Transient Analysis", False):
                st.subheader("Approximate Transient Analysis:")
                fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Ib'],name='Ib'), row=1, col=1)
                fig.add_vline(x=0.501, line_width=3, line_dash="dash", line_color="red")
                fig.add_annotation(x=0.57, y=1.989, text="Subtransient", showarrow=False, arrowhead=1)
                fig.add_vline(x=0.64, line_width=3, line_dash="dash", line_color="black")
                fig.add_annotation(x=0.75, y=1.989, text="Transient", showarrow=False, arrowhead=1)
                fig.add_vline(x=0.85675, line_width=3, line_dash="dash", line_color="green")
                fig.add_annotation(x=0.928, y=1.989, text="Steady State", showarrow=False, arrowhead=1)
            else:
                fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Ib'],name='Ib'), row=1, col=1)
                fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Vb'],name='Vb'), row=2, col=1)
                
                
            # Update xaxis properties
            fig.update_xaxes(title_text="Amp", row=1, col=1)
            fig.update_xaxes(title_text="Volts", row=2, col=1)


            # Update yaxis properties
            fig.update_yaxes(title_text="Time", row=1, col=1)
            fig.update_yaxes(title_text="Time", row=2, col=1)


            # Update title and height
            fig.update_layout(title_text="Phase B Data Visualizations (Rf = 0.1 Ohms, TL Length = 300 Km)", height=700)
            st.plotly_chart(fig,use_container_width=True) 

        if option2 == 'Phase: C': 
    #         fig = make_subplots(rows=2, cols=1)
    #         fig.append_trace(go.Scatter(x=f['Domain'],y=f[' Vc'],name='Vc (Volts)'), row=1, col=1)
    #         fig.update_layout(title_text="Phase C Data Visualizations (Rf = 0.1 Ohms, TL Length = 300 Km)")
    #         fig.append_trace(go.Scatter(x=f['Domain'],y=f[' Ic'],name='Ic (Amps)'), row=2, col=1)
    #         fig.update_layout(yaxis_title='Time ')
    #         st.plotly_chart(fig,use_container_width=True)

            # Initialize figure with subplots
            fig = make_subplots(
                rows=2, cols=1 
                #,subplot_titles=("Va vs Time", "Ia Vs Time")
            )

                        # Add traces
            if st.sidebar.checkbox("Show Transient Analysis", False):
                st.subheader("Approximate Transient Analysis:")
                fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Ic'],name='Ic'), row=1, col=1)
                fig.add_vline(x=0.501, line_width=3, line_dash="dash", line_color="red")
                fig.add_annotation(x=0.57, y=1.989, text="Subtransient", showarrow=False, arrowhead=1)
                fig.add_vline(x=0.64, line_width=3, line_dash="dash", line_color="black")
                fig.add_annotation(x=0.75, y=1.989, text="Transient", showarrow=False, arrowhead=1)
                fig.add_vline(x=0.85675, line_width=3, line_dash="dash", line_color="green")
                fig.add_annotation(x=0.928, y=1.989, text="Steady State", showarrow=False, arrowhead=1)
            else:
                fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Ic'],name='Ic'), row=1, col=1)
                fig.add_trace(go.Scatter(x=f['Domain'],y=f[' Vc'],name='Vc'), row=2, col=1)
                
                
            # Update xaxis properties
            fig.update_xaxes(title_text="Amp", row=1, col=1)
            fig.update_xaxes(title_text="Volts", row=2, col=1)


            # Update yaxis properties
            fig.update_yaxes(title_text="Time", row=1, col=1)
            fig.update_yaxes(title_text="Time", row=2, col=1)


            # Update title and height
            fig.update_layout(title_text="Phase C Data Visualizations (Rf = 0.1 Ohms, TL Length = 300 Km)", height=700)
            st.plotly_chart(fig,use_container_width=True)

    #########################################################################################################################################3
    if option=="0.001 Ohm":

        AB_0_001=pd.read_csv("AB_0.001.csv")
        ABC_0_001=pd.read_csv("ABC_0.001.csv")
        ABG_0_001=pd.read_csv("ABG_0.001.csv")
        AC_0_001=pd.read_csv("AC_0.001.csv")
        ACG_0_001=pd.read_csv("ACG_0.001.csv")
        AG_0_001=pd.read_csv("AG_0.001.csv")
        BC_0_001=pd.read_csv("BC_0.001.csv")
        BCG_0_001=pd.read_csv("BCG_0.001.csv")
        BG_0_001=pd.read_csv("BG_0.001.csv")
        CG_0_001=pd.read_csv("CG_0.001.csv")
        No_0_001=pd.read_csv("No_Fault_0.001.csv")

        x1 = st.sidebar.selectbox('Choose No Fault or Fault Type',('No Fault','AB','ABC','ABG','AC','ACG','AG','BC','BCG','BG','CG'))

        if x1=='AB':
            f1=AB_0_001
            st.subheader("AB Fault Analysis:")
        elif x1=='ABC':
            f1=ABC_0_001
            st.subheader("ABC Fault Analysis:")
        elif x1=='ABG':
            f1=ABG_0_001
            st.subheader("ABG Fault Analysis:")
        elif x1=='AC':
            f1=AC_0_001
            st.subheader("AC Fault Analysis:")
        elif x1=='ACG':
            f1=ACG_0_001
            st.subheader("ACG Fault Analysis:")
        elif x1=='AG':
            f1=AG_0_001
            st.subheader("AG Fault Analysis:")
        elif x1=='BC':
            f1=BC_0_001
            st.subheader("BC Fault Analysis:")
        elif x1=='BCG':
            f1=BCG_0_001
            st.subheader("BCG Fault Analysis:")
        elif x1=='BG':
            f1=BG_0_001
            st.subheader("BG Fault Analysis:")
        elif x1=='CG':
            f1=CG_0_001
            st.subheader("CG Fault Analysis:")
        elif x1=='No Fault':
            f1=No_0_001
            st.subheader("No Fault Condition:")

        fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Va'],name='Va'))
        fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Vb'],name='Vb'))
        fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Vc'],name='Vc'))
        fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Ia'],name='Ia'))
        fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Ib'],name='Ib'))
        fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Ic'],name='Ic'))
        fig.update_layout(title_text="Combined Data Visualizations (Rf = 0.001 Ohms, TL Length = 300 Km)")
        fig.update_layout(xaxis_title='3-Phase Voltages (Volts) & Currents (Amp)')
        fig.update_layout(yaxis_title='Time (Sec)')
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig,use_container_width=True)    
    ###############################################################################################################################    
        option3 = st.sidebar.radio('Choose Phases (A, B or C)',['Phase: A','Phase: B','Phase: C'])

        if option3 == 'Phase: A':
    #         fig = make_subplots(rows=2, cols=1)
    #         fig.append_trace(go.Scatter(x=f1['Domain'],y=f1['Va'],name='Va (Volts)'), row=1, col=1)
    #         fig.update_layout(title_text="Phase A Data Visualizations (Rf = 0.001 Ohms, TL Length = 300 Km)")
    #         fig.append_trace(go.Scatter(x=f1['Domain'],y=f1['Ia'],name='Ia (Amps)'), row=2, col=1)
    #         fig.update_layout(yaxis_title='Time ')
    #         st.plotly_chart(fig,use_container_width=True)

            # Initialize figure with subplots
            fig = make_subplots(
                rows=2, cols=1 
                #,subplot_titles=("Va vs Time", "Ia Vs Time")
            )

            # Add traces
            if st.sidebar.checkbox("Show Transient Analysis", False):
                st.subheader("Approximate Transient Analysis:")
                fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Ia'],name='Ia'), row=1, col=1)
                fig.add_vline(x=0.501, line_width=3, line_dash="dash", line_color="red")
                fig.add_annotation(x=0.57, y=1.989, text="Subtransient", showarrow=False, arrowhead=1)
                fig.add_vline(x=0.64, line_width=3, line_dash="dash", line_color="black")
                fig.add_annotation(x=0.75, y=1.989, text="Transient", showarrow=False, arrowhead=1)
                fig.add_vline(x=0.85675, line_width=3, line_dash="dash", line_color="green")
                fig.add_annotation(x=0.928, y=1.989, text="Steady State", showarrow=False, arrowhead=1)
            else:
                fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Ia'],name='Ia'), row=1, col=1)
                fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Va'],name='Va'), row=2, col=1)
                
                
            # Update xaxis properties
            fig.update_xaxes(title_text="Amp", row=1, col=1)
            fig.update_xaxes(title_text="Volts", row=2, col=1)


            # Update yaxis properties
            fig.update_yaxes(title_text="Time (Sec)", row=1, col=1)
            fig.update_yaxes(title_text="Time (Sec)", row=2, col=1)


            # Update title and height
            fig.update_layout(title_text="Phase A Data Visualizations (Rf = 0.001 Ohms, TL Length = 300 Km)", height=700)
            st.plotly_chart(fig,use_container_width=True)

        if option3 == 'Phase: B': 
    #         fig = make_subplots(rows=2, cols=1)
    #         fig.append_trace(go.Scatter(x=f1['Domain'],y=f1['Vb'],name='Vb (Volts)'), row=1, col=1)
    #         fig.update_layout(title_text="Phase B Data Visualizations (Rf = 0.001 Ohms, TL Length = 300 Km)")
    #         fig.append_trace(go.Scatter(x=f1['Domain'],y=f1['Ib'],name='Ib (Amps)'), row=2, col=1)
    #         fig.update_layout(yaxis_title='Time ')
    #         st.plotly_chart(fig,use_container_width=True)
            # Initialize figure with subplots
            fig = make_subplots(
                rows=2, cols=1 
                #,subplot_titles=("Va vs Time", "Ia Vs Time")
            )

            # Add traces
            if st.sidebar.checkbox("Show Transient Analysis", False):
                st.subheader("Approximate Transient Analysis:")
                fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Ib'],name='Ib'), row=1, col=1)
                fig.add_vline(x=0.501, line_width=3, line_dash="dash", line_color="red")
                fig.add_annotation(x=0.57, y=1.989, text="Subtransient", showarrow=False, arrowhead=1)
                fig.add_vline(x=0.64, line_width=3, line_dash="dash", line_color="black")
                fig.add_annotation(x=0.75, y=1.989, text="Transient", showarrow=False, arrowhead=1)
                fig.add_vline(x=0.85675, line_width=3, line_dash="dash", line_color="green")
                fig.add_annotation(x=0.928, y=1.989, text="Steady State", showarrow=False, arrowhead=1)
            else:
                fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Ib'],name='Ib'), row=1, col=1)
                fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Vb'],name='Vb'), row=2, col=1)
                
                
            # Update xaxis properties
            fig.update_xaxes(title_text="Amp", row=1, col=1)
            fig.update_xaxes(title_text="Volts", row=2, col=1)


            # Update yaxis properties
            fig.update_yaxes(title_text="Time (Sec)", row=1, col=1)
            fig.update_yaxes(title_text="Time (Sec)", row=2, col=1)


            # Update title and height
            fig.update_layout(title_text="Phase B Data Visualizations (Rf = 0.001 Ohms, TL Length = 300 Km)", height=700)
            st.plotly_chart(fig,use_container_width=True)

        if option3 == 'Phase: C': 
    #         fig = make_subplots(rows=2, cols=1)
    #         fig.append_trace(go.Scatter(x=f1['Domain'],y=f1['Vc'],name='Vc (Volts)'), row=1, col=1)
    #         fig.update_layout(title_text="Phase C Data Visualizations (Rf = 0.001 Ohms, TL Length = 300 Km)")
    #         fig.append_trace(go.Scatter(x=f1['Domain'],y=f1['Ic'],name='Ic (Amps)'), row=2, col=1)
    #         fig.update_layout(yaxis_title='Time ')
    #         st.plotly_chart(fig,use_container_width=True)
            # Initialize figure with subplots
            fig = make_subplots(
                rows=2, cols=1 
                #,subplot_titles=("Va vs Time", "Ia Vs Time")
            )

            # Add traces
            if st.sidebar.checkbox("Show Transient Analysis", False):
                st.subheader("Approximate Transient Analysis:")
                fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Ic'],name='Ic'), row=1, col=1)
                fig.add_vline(x=0.501, line_width=3, line_dash="dash", line_color="red")
                fig.add_annotation(x=0.57, y=1.989, text="Subtransient", showarrow=False, arrowhead=1)
                fig.add_vline(x=0.64, line_width=3, line_dash="dash", line_color="black")
                fig.add_annotation(x=0.75, y=1.989, text="Transient", showarrow=False, arrowhead=1)
                fig.add_vline(x=0.85675, line_width=3, line_dash="dash", line_color="green")
                fig.add_annotation(x=0.928, y=1.989, text="Steady State", showarrow=False, arrowhead=1)
            else:
                fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Ic'],name='Ic'), row=1, col=1)
                fig.add_trace(go.Scatter(x=f1['Domain'],y=f1['Vc'],name='Vc'), row=2, col=1)
                
                
            # Update xaxis properties
            fig.update_xaxes(title_text="Amp", row=1, col=1)
            fig.update_xaxes(title_text="Volts", row=2, col=1)


            # Update yaxis properties
            fig.update_yaxes(title_text="Time (Sec)", row=1, col=1)
            fig.update_yaxes(title_text="Time (Sec)", row=2, col=1)


            # Update title and height
            fig.update_layout(title_text="Phase C Data Visualizations (Rf = 0.001 Ohms, TL Length = 300 Km)", height=700)
            st.plotly_chart(fig,use_container_width=True)
            
################################################################# 3rd selection ######################################################       ################################################################# 3rd selection ######################################################  
################################################################# 3rd selection ######################################################  
################################################################# 3rd selection ######################################################  
################################################################# 3rd selection ###################################################### 
                          
elif Side=="Prediction Results":
    data_load_state = st.text('Loading data...')
    p2=pd.read_csv('fault_results.csv')
    p3=pd.read_csv('fault_labels.csv')
    data_load_state.text("")

    if st.sidebar.checkbox("Display Labels", False):
        st.sidebar.write("Fault/ No Fault Labels:")
        st.sidebar.write(p3)

    classifier = st.sidebar.selectbox("Choose Classifier Results", ("All ML Classification Models", "K Neighbors Classifier"))

    if classifier=="All ML Classification Models":
        st.subheader("All Machine Learning Classification Models Results:")
        st.write(p2)

    elif classifier=="K Neighbors Classifier":
        st.subheader("K Neighbors Classifier Results:")
        option2 = st.sidebar.radio('Ploting Performance Parameters:',['Confusion Matrix','Error','Model Class_report','Validation Curve'])

        if option2=="Confusion Matrix":
            image = Image.open('KNN_CM.png')
            st.image(image, caption='')

        elif option2=="Error":
            image = Image.open('KNN_ERROR.png')
            st.image(image, caption='')

        elif option2=="Model Class_report":
            image = Image.open('KNN_CR.png')
            st.image(image, caption='')

        elif option2=="Validation Curve":
            image = Image.open('KNN_VC.png')
            st.image(image, caption='')
    
    
    
################################################################# 4th selection ######################################################       ################################################################# 4th selection ######################################################  
################################################################# 4th selection ######################################################  
################################################################# 4th selection ######################################################  
################################################################# 4th selection ######################################################  
elif Side=='Real Time Predictions':
    
    data_load_state = st.text('Loading data...')
    p=pd.read_csv('COMPACT_0.001.csv')
    p=p.sample(frac=1)
    s=setup(p,target="Fault",train_size=0.8)
    data_load_state.text("")
    st.subheader("K Neighbors Classifier Real Time Predictions:")
    # if st.sidebar.checkbox("Display data", False):
    #     st.subheader("Show Fault Dataset")
    #     st.write(p)
    st.sidebar.write("Enter Voltages and Currents")
    Va=st.sidebar.number_input('Va (Volts)')
    Vb=st.sidebar.number_input('Vb (Volts)')
    Vc=st.sidebar.number_input('Vc (Volts)') 
    Ia=st.sidebar.number_input('Ia (Amperes)')
    Ib=st.sidebar.number_input('Ib (Amperes)')
    Ic=st.sidebar.number_input('Ic (Amperes)')
    model=create_model("knn")
        
    df=pd.DataFrame([[Va,Vb,Vc,Ia,Ib,Ic]],columns=["Va","Vb","Vc","Ia","Ib","Ic"])
    st.write("Inputs:")
    df
    if st.button ('Predict'):
        st.write("Predictions:")
        f=predict_model(model,df)
        f
        audio_file = open('audio.ogg', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')

                          
                          
                          
                          
                          
                          
                          
                          
