import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import re
import numpy as np



all_data = {}

# Assuming the data is in the following format as per the given image:
all_data['uci'] = """
0.6314+-0.0145	0.6496+-0.0062	0.6075+-0.0201	0.6095+-0.0162	0.6116+-0.0255
0.5814+-0.0069	0.5600+-0.0063	0.5659+-0.0166	0.5468+-0.0118	0.5203+-0.0159
0.3618+-0.0125	0.4158+-0.0157	0.4532+-0.0269	0.5204+-0.0343	0.5352+-0.0265
0.2710+-0.0078	0.3132+-0.0098	0.3760+-0.0120	0.4669+-0.0132	0.5103+-0.0054
0.6555+-0.0131	0.6081+-0.0202	0.5508+-0.0435	0.5349+-0.0302	0.5630+-0.0126
0.5977+-0.0136	0.5580+-0.0416	0.5161+-0.0258	0.4740+-0.0229	0.4511+-0.0241
0.2900+-0.0498	0.3231+-0.0541	0.3382+-0.0170	0.4180+-0.0240	0.4452+-0.0115
0.2392+-0.0333	0.2651+-0.0187	0.2682+-0.0195	0.2968+-0.0203	0.3330+-0.0179
"""
all_data['CollegeMsg'] = """
0.6219+-0.0571	0.5955+-0.0300	0.6237+-0.0228	0.6112+-0.0233	0.6065+-0.0306
0.5892+-0.0115	0.5654+-0.0138	0.5575+-0.0088	0.5496+-0.0126	0.5429+-0.0120
0.4001+-0.0571	0.4411+-0.0190	0.4758+-0.0132	0.5210+-0.0194	0.5475+-0.0232
0.2680+-0.0068	0.3058+-0.0112	0.3610+-0.0154	0.4611+-0.0235	0.5011+-0.0100
0.6494+-0.0097	0.6032+-0.0275	0.5716+-0.0357	0.5535+-0.0098	0.5587+-0.0164
0.6121+-0.0234	0.5777+-0.0205	0.5176+-0.0171	0.4824+-0.0228	0.4397+-0.0106
0.2886+-0.0181	0.2884+-0.0507	0.3376+-0.0292	0.4272+-0.0259	0.4681+-0.0177
0.2504+-0.0174	0.2541+-0.0096	0.2526+-0.0123	0.2894+-0.0206	0.3277+-0.0170
"""
all_data['WIKI'] = """
0.8618+-0.0046	0.8577+-0.0037	0.8617+-0.0042	0.8579+-0.0040	0.8556+-0.0031
0.8397+-0.0053	0.8438+-0.0019	0.8372+-0.0036	0.8357+-0.0065	0.8252+-0.0042
0.7992+-0.0104	0.8081+-0.0051	0.8107+-0.0082	0.8212+-0.0055	0.8387+-0.0073
0.7781+-0.0071	0.7929+-0.0035	0.7975+-0.0042	0.8099+-0.0035	0.8139+-0.0040
0.8416+-0.0063	0.8447+-0.0052	0.8484+-0.0021	0.8381+-0.0103	0.8160+-0.0033
0.7102+-0.0045	0.7098+-0.0084	0.7112+-0.0085	0.7120+-0.0085	0.6829+-0.0081
0.7057+-0.0065	0.7170+-0.0140	0.7380+-0.0080	0.7460+-0.0138	0.7541+-0.0162
0.6872+-0.0126	0.6804+-0.0185	0.6481+-0.0108	0.6421+-0.0132	0.6393+-0.0122
"""
all_data['REDDIT'] = """
0.8146+-0.0096	0.8138+-0.0146	0.8138+-0.0119	0.8193+-0.0147	0.8169+-0.0118
0.8536+-0.0012	0.8566+-0.0014	0.8570+-0.0016	0.8562+-0.0031	0.8544+-0.0022
0.7093+-0.0342	0.7194+-0.0216	0.7221+-0.0350	0.7287+-0.0149	0.7436+-0.0312
0.8475+-0.0015	0.8511+-0.0012	0.8540+-0.0020	0.8515+-0.0018	0.8516+-0.0016
0.7963+-0.0088	0.8022+-0.0090	0.8297+-0.0116	0.8402+-0.0132	0.8389+-0.0044
0.7871+-0.0010	0.8086+-0.0028	0.8244+-0.0012	0.8322+-0.0021	0.8281+-0.0009
0.6414+-0.0354	0.6923+-0.0279	0.7384+-0.0135	0.7627+-0.0313	0.7365+-0.0233
0.8161+-0.0011	0.8337+-0.0018	0.8427+-0.0022	0.8397+-0.0019	0.8286+-0.0025
"""
all_data['Flights'] = """
0.4585+-0.0382	0.5400+-0.0399	0.5035+-0.0210	0.5194+-0.0214	0.5843+-0.0719
0.8275+-0.0014	0.8273+-0.0014	0.8326+-0.0012	0.8328+-0.0010	0.8322+-0.0021
0.4487+-0.0648	0.3902+-0.1427	0.4648+-0.0578	0.3668+-0.0556	0.4371+-0.0513
0.8296+-0.0034	0.8315+-0.0020	0.8304+-0.0014	0.8284+-0.0023	0.8300+-0.0011
0.4431+-0.0411	0.4559+-0.0895	0.5198+-0.0243	0.5268+-0.0282	0.5394+-0.0260
0.7440+-0.0007	0.7804+-0.0008	0.8040+-0.0021	0.8188+-0.0019	0.8217+-0.0026
0.2566+-0.0702	0.3397+-0.0450	0.3544+-0.0879	0.4235+-0.0278	0.4126+-0.0467
0.7246+-0.0048	0.7776+-0.0040	0.7988+-0.0025	0.8133+-0.0022	0.8129+-0.0033
"""
all_data['LASTFM'] = """
0.4585+-0.0382	0.5400+-0.0399	0.5035+-0.0210	0.5194+-0.0214	0.5843+-0.0719
0.8275+-0.0014	0.8273+-0.0014	0.8326+-0.0012	0.8328+-0.0010	0.8322+-0.0021
0.4487+-0.0648	0.3902+-0.1427	0.4648+-0.0578	0.3668+-0.0556	0.4371+-0.0513
0.8296+-0.0034	0.8315+-0.0020	0.8304+-0.0014	0.8284+-0.0023	0.8300+-0.0011
0.4431+-0.0411	0.4559+-0.0895	0.5198+-0.0243	0.5268+-0.0282	0.5394+-0.0260
0.7440+-0.0007	0.7804+-0.0008	0.8040+-0.0021	0.8188+-0.0019	0.8217+-0.0026
0.2566+-0.0702	0.3397+-0.0450	0.3544+-0.0879	0.4235+-0.0278	0.4126+-0.0467
0.7246+-0.0048	0.7776+-0.0040	0.7988+-0.0025	0.8133+-0.0022	0.8129+-0.0033
"""

# Set distinguishable green colors for the first four columns and red colors for the latter four
green_colors = ['#98FB98', '#6B8E23', '#008080', '#00441b']  # Shades of green
red_colors = ['#FF9999', '#fb6a4a', '#cb181d', '#990000']  # Shades of red
colors = green_colors + red_colors

for dataset in ['uci', 'CollegeMsg', 'WIKI', 'REDDIT', 'Flights', 'LASTFM']:
    data = all_data[dataset]
    
    # Split the data into rows and then into individual cell values
    rows = data.strip().split("\n")
    split_data = [re.split(r"\t+", row) for row in rows]

    # Extract mean and standard deviation from each cell value
    mean_std_data = []
    for row in split_data:
        row_data = []
        for cell in row:
            mean, std = map(float, cell.split('+-'))
            row_data.append((mean, std))
        mean_std_data.append(row_data)

    # Convert the mean and standard deviation data into a DataFrame
    df = pd.DataFrame(mean_std_data, columns=[f"Data {i+1}" for i in range(len(mean_std_data[0]))]).T
    df.columns = [
        "att + gru + re",
        "att + embed + re",
        "att + gru + uni",
        "att + embed + uni",
        "mixer + gru + re",
        "mixer + embed + re",
        "mixer + gru + uni",
        "mixer + embed + uni"
    ]
    # Now let's create a figure with error bars representing the standard deviation
    plt.figure(figsize=(14, 8))

    # Plot each column's data
    for i, col in enumerate(df.columns):
        means = df[col].apply(lambda x: x[0])
        stds = df[col].apply(lambda x: x[1])
        plt.errorbar(x=range(len(means)), y=means, yerr=stds, fmt='-o', capsize=5, label=col, color=colors[i])

    plt.title(f'{dataset} MRR with Different Neighbor Count')
    x_labels = [5, 10, 20, 50, 100]
    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)
    # plt.xlabel('Index')
    # plt.ylabel('Mean Value')
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO object and return its content
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_data = buf.read()
    buf.close()

    # Save the image data to a file
    output_path = f'{dataset}_mean_std_plot.png'
    with open(output_path, 'wb') as f:
        f.write(image_data)

    output_path