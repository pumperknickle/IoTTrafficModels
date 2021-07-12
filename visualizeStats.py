import matplotlib.pyplot as plt
import pickle
import numpy as np
import plotly.graph_objects as go

results_file = "average_packet_lengths_iotg.pkl"
with open(results_file, mode='rb') as tokenFile:
    average_packet_lengths_iotg = pickle.load(tokenFile)

results_file = "average_packet_lengths_real.pkl"
with open(results_file, mode='rb') as tokenFile:
    average_packet_lengths_real = pickle.load(tokenFile)

results_file = "stdev_packet_lengths_iotg.pkl"
with open(results_file, mode='rb') as tokenFile:
    stdev_packet_lengths_iotg = pickle.load(tokenFile)

results_file = "stdev_packet_lengths_real.pkl"
with open(results_file, mode='rb') as tokenFile:
    stdev_packet_lengths_real = pickle.load(tokenFile)

results_file = "average_duration_iotg.pkl"
with open(results_file, mode='rb') as tokenFile:
    average_duration_iotg = pickle.load(tokenFile)

results_file = "average_duration_real.pkl"
with open(results_file, mode='rb') as tokenFile:
    average_duration_real = pickle.load(tokenFile)

results_file = "stdev_duration_iotg.pkl"
with open(results_file, mode='rb') as tokenFile:
    stdev_duration_iotg = pickle.load(tokenFile)

results_file = "stdev_duration_real.pkl"
with open(results_file, mode='rb') as tokenFile:
    stdev_duration_real = pickle.load(tokenFile)

results_file = "average_packet_lengths_dg.pkl"
with open(results_file, mode='rb') as tokenFile:
    average_packet_lengths_dg = pickle.load(tokenFile)

results_file = "stdev_packet_lengths_dg.pkl"
with open(results_file, mode='rb') as tokenFile:
    stdev_packet_lengths_dg = pickle.load(tokenFile)

results_file = "average_duration_dg.pkl"
with open(results_file, mode='rb') as tokenFile:
    average_duration_dg = pickle.load(tokenFile)

results_file = "stdev_duration_dg.pkl"
with open(results_file, mode='rb') as tokenFile:
    stdev_duration_dg = pickle.load(tokenFile)

devices_labels = []
a_p_l_i = []
a_p_l_r = []
s_p_l_i = []
s_p_l_r = []
a_d_i = []
a_d_r = []
s_d_i = []
s_d_r = []
a_p_l_d = []
s_p_l_d = []
a_d_d = []
s_d_d = []

for key in average_duration_iotg.keys():
    devices_labels.append(key[16:][:-8])
    a_p_l_i.append((abs(average_packet_lengths_iotg[key] - average_packet_lengths_real[key])/(average_packet_lengths_real[key])) * 100)
    s_p_l_i.append((abs(stdev_packet_lengths_iotg[key] - stdev_packet_lengths_real[key])/(stdev_packet_lengths_real[key])) * 100)
    a_d_i.append((abs(average_duration_iotg[key] - average_duration_real[key])/average_duration_real[key]) * 100)
    s_d_i.append((abs(stdev_duration_iotg[key] - stdev_duration_real[key])/stdev_duration_real[key]) * 100)
    a_p_l_d.append((abs(average_packet_lengths_dg[key] - average_packet_lengths_real[key])/average_packet_lengths_real[key]) * 100)
    s_p_l_d.append((abs(stdev_packet_lengths_dg[key] - stdev_packet_lengths_real[key])/stdev_packet_lengths_real[key]) * 100)
    a_d_d.append((abs(average_duration_dg[key] - average_duration_real[key])/ average_duration_real[key]) * 100)
    s_d_d.append((abs(stdev_duration_dg[key] - stdev_duration_real[key])/stdev_duration_real[key]) * 100)

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}]],
    subplot_titles=("Average Packet Length (% Relative Error)","Standard Deviation Packet Length (% Relative Error)", "Average Duration Between Packets (% Relative Error)", "Standard Deviation Duration Between Packets (% Relative Error)")
)

fig.add_trace(go.Bar(
    x=devices_labels,
    y=a_p_l_i,
    name='IoTGenerator',
    marker_color='indianred'
), row=1,
    col=1)

fig.add_trace(go.Bar(
    x=devices_labels,
    y=a_p_l_d,
    name='DoppelGANger',
    marker_color='green'
), row=1,
    col=1)

fig.add_trace(go.Bar(
    x=devices_labels,
    y=s_p_l_i,
    name='IoTGenerator',
    marker_color='indianred'
), row=1,
    col=2)

fig.add_trace(go.Bar(
    x=devices_labels,
    y=s_p_l_d,
    name='DoppelGANger',
    marker_color='green'
),
    row=1,
    col=2)

fig.add_trace(go.Bar(
    x=devices_labels,
    y=a_d_i,
    name='IoTGenerator',
    marker_color='indianred'
),
    row=2,
    col=1)

fig.add_trace(go.Bar(
    x=devices_labels,
    y=a_d_d,
    name='DoppelGANger',
    marker_color='green'
),
    row=2,
    col=1)

fig.add_trace(go.Bar(
    x=devices_labels,
    y=s_d_i,
    name='IoTGenerator',
    marker_color='indianred'
),
    row=2,
    col=2)

fig.add_trace(go.Bar(
    x=devices_labels,
    y=s_d_d,
    name='DoppelGANger',
    marker_color='green'
),
    row=2,
    col=2)


fig.update_layout(barmode='group')
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()