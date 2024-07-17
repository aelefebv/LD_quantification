import pandas as pd
import os
import matplotlib.pyplot as plt


file_dir = r'path\to\files'

intensity_cutoff = 4000
dataset_merge_path = rf"{file_dir}\file_name_{intensity_cutoff}_mapped.csv"

# Filter the dataset by intensity
if os.path.exists(dataset_merge_path):
    print(f'Loading dataset intensity filtered mapped file')
    dataset_df_merge = pd.read_csv(dataset_merge_path)
else:
    print(f'Creating dataset intensity filtered mapped file')
    dataset_map = pd.read_csv(rf"{file_dir}\data-platemap.csv")
    temp_df = pd.read_csv(fr"{file_dir}\processing_output-ld_stats.csv")
    dataset_df = temp_df[temp_df['ld_intensity'] > intensity_cutoff]
    dataset_df['destination_well'] = dataset_df['row'].astype(str) + dataset_df['column'].astype(str).str.zfill(2)
    dataset_df_merge = dataset_map.merge(dataset_df, on='destination_well')
    dataset_df_merge.to_csv(dataset_merge_path)

df_names = ['df_name_1', 'df_name_2']
for df_num, df in enumerate([dataset_df_merge, dataset2_df_merge]):
    all_grouped = df.groupby(['destination_well', 'timepoint', 'treatment', 'dimerizer_concentration', 'drug', 'drug_conc_mM'])
    all_mean = all_grouped['ld_area'].mean()
    all_mean = all_mean.reset_index()
    all_mean_df = pd.DataFrame(all_mean)

    grouped_by_treatment = all_mean_df.groupby(['treatment', 'timepoint', 'dimerizer_concentration', 'drug', 'drug_conc_mM'])
    mean_by_treatment = grouped_by_treatment['ld_area'].mean()
    mean_by_treatment = mean_by_treatment.reset_index()
    mean_by_treatment_df = pd.DataFrame(mean_by_treatment)

    # get std
    std_by_treatment = grouped_by_treatment['ld_area'].std()
    std_by_treatment = std_by_treatment.reset_index()
    std_by_treatment_df = pd.DataFrame(std_by_treatment)

    # Create a dictionary to assign a unique color to each drug
    drug_colors = {
        'drug1': 'gray',
        'drug2': 'red',
        'drug3': 'blue',
        'drug4': 'green',
        'drug5': 'orange',
        'drug6': 'purple',
        # Add more drugs as needed
    }

    # Create a plot
    plt.figure(figsize=(12, 6))
    # Iterate over each unique value of dimerizer concentration
    for dimerizer_conc in mean_by_treatment_df['dimerizer_concentration'].unique():
        # Filter the DataFrame for the current AP20187_conc_uM value
        filtered_data = mean_by_treatment_df[mean_by_treatment_df['dimerizer_concentration'] == dimerizer_conc]
        filtered_std = std_by_treatment_df[std_by_treatment_df['dimerizer_concentration'] == dimerizer_conc]

        # Create a plot for each AP20187_conc_uM value
        plt.figure(figsize=(12, 6))

        # Iterate over each unique treatment
        for treatment in filtered_data['treatment'].unique():
            treatment_data = filtered_data[filtered_data['treatment'] == treatment]
            treatment_std = filtered_std[filtered_std['treatment'] == treatment]

            # Get the drug and its concentration for this treatment
            drug = treatment_data['drug'].iloc[0]
            concentration = treatment_data['drug_conc_mM'].iloc[0]

            # Get the base color for the drug
            base_color = drug_colors.get(drug, 'gray')  # Default to gray if drug not in dictionary

            # Use a colormap to get a shade based on the intensity
            shade_intensity = 0.8
            color = plt.cm.Blues(shade_intensity) if base_color == 'blue' else \
                    plt.cm.Reds(shade_intensity) if base_color == 'red' else \
                    plt.cm.Greens(shade_intensity) if base_color == 'green' else \
                    plt.cm.Oranges(shade_intensity) if base_color == 'orange' else \
                    plt.cm.Purples(shade_intensity) if base_color == 'purple' else \
                    plt.cm.Greys(shade_intensity) if base_color == 'gray' else \
                    base_color

            # Plot mean with std
            plt.errorbar(treatment_data['timepoint'], treatment_data['ld_area'], yerr=treatment_std['ld_area'],
                         label=treatment, color=color)

        # Adding plot title and labels
        plt.title(f'LD Area over Time by Treatment (AP20187_conc_uM: {dimerizer_conc})')
        plt.xlabel('Timepoint')
        plt.ylabel('LD Area')
        plt.legend()
        plt.ylim(5, 30)

        # plt.show()
        # Save the plot
        im_name = f'{df_names[df_num]}-ld_area_time_by_treatment-dimerizer_concentration_{dimerizer_conc}.png'
        plt.savefig(os.path.join(file_dir, im_name))
        plt.close()
