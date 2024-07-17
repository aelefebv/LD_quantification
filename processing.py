import glob
import re
import os
from dataclasses import dataclass
import cupyx.scipy.ndimage as ndi
import scipy.spatial as spatial
import cupy as xp
import numpy as np
import tifffile
import skimage.filters
import scipy.ndimage
import skimage.morphology
from csbdeep.utils import normalize
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')
viewer = None
import datetime

dt = datetime.datetime.now()
# YYYYMMDD_HHMMSS
dt_str = dt.strftime("%Y%m%d_%H%M%S")
dt_hms = dt.strftime("%H:%M:%S")


@dataclass
class LD:
    """
    Dataclass to hold information about a single LD
    """
    row: str
    column: int
    field: int
    timepoint: int
    combo_name: str = None
    num_nuc: int = None
    ld_radius: float = None
    ld_intensity: float = None
    ld_area: float = None

class MultiScaleSettings:
    """Class to hold settings for multiscale processing"""
    def __init__(self, min_radius_px=2, max_radius_px=8, num_sigma=5, min_sigma_step_size=0.2):
        """
        :param min_radius_px: Minimum radius in pixels
        :param max_radius_px: Maximum radius in pixels
        :param num_sigma: Number of sigma values to use
        :param min_sigma_step_size: Minimum sigma step size
        """
        self.min_sigma_step_size = min_sigma_step_size
        self.num_sigma = num_sigma

        self.sigma_min = min_radius_px / 2
        self.sigma_max = max_radius_px / 3

        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / self.num_sigma
        sigma_step_size = max(self.min_sigma_step_size, sigma_step_size_calculated)  # Avoid taking too small of steps.

        self.sigmas = list(xp.arange(self.sigma_min, self.sigma_max, sigma_step_size))

def get_max_intensity_timelapse(im_dict, im_size=1080):
    """
    Get the maximum intensity projection of a timelapse
    :param im_dict: Dictionary of timepoints with list of planes
    """
    mp_im = np.zeros((len(im_dict), im_size, im_size), dtype=np.uint16)
    for i, (timepoint, planes) in enumerate(im_dict.items()):
        max_intensity_plane = np.zeros((im_size, im_size), dtype=np.uint16)
        for plane, file in planes:
            mp_im[i] = np.maximum(max_intensity_plane, tifffile.imread(file))
    return mp_im

def run_frame(frame_num, ld_frame, nuc_frame, ms_set):
    """
    Run the frame
    :param frame_num: Frame number
    :param ld_frame: LD frame
    :param nuc_frame: Nucleus frame
    :param ms_set: MultiScaleSettings object
    :return: List of LD objects
    """
    # Initialize the LD object
    ld_frame_info = LD(row=run_group[0], column=run_group[1], field=run_group[2], timepoint=frame_num,
                       combo_name=f'{run_group[0]}_{run_group[1]}_{run_group[2]}')

    # Get nuclear labels
    nuc_plane = normalize(nuc_frame)
    nuc_labels, _ = model.predict_instances(nuc_plane)
    nuc_labels = skimage.morphology.remove_small_objects(nuc_labels, min_size=1000)
    ld_frame_info.num_nuc = len(np.unique(nuc_labels)) - 1

    # Process the LD frame
    current_frame = xp.asarray(ld_frame).astype(float)
    big_gauss = ndi.gaussian_filter(current_frame, sigma=2)
    thresh = skimage.filters.threshold_triangle(big_gauss.get()) / 2
    big_mask = xp.asarray(big_gauss > thresh)

    # Run the multiscale processing
    lapofg = np.empty(((len(ms_set.sigmas),) + current_frame.shape), dtype=float)
    for i, s in enumerate(ms_set.sigmas):
        current_lapofg = -ndi.gaussian_laplace(current_frame, s) * xp.mean(s) ** 2
        current_lapofg[~big_mask] = 0
        current_lapofg[current_lapofg < 0] = 0
        lapofg[i] = current_lapofg.get()

    lapofg_thresh = skimage.filters.threshold_triangle(lapofg[0][lapofg[0] > 0]) / 2
    ld_mask = xp.asarray(lapofg[0] > lapofg_thresh) * big_mask
    dilated_mask = ndi.binary_dilation(ld_mask)

    # Run a maximum filter to get the peaks
    filt_footprint = np.ones((3,) * (current_frame.ndim + 1))
    max_filt = scipy.ndimage.maximum_filter(lapofg, footprint=filt_footprint, mode='nearest')
    peaks = xp.empty(lapofg.shape, dtype=bool)
    for z_slice, max_filt_slice in enumerate(max_filt):
        peaks[z_slice] = (xp.asarray(lapofg[z_slice]) == xp.asarray(max_filt_slice)) * ld_mask

    # Get the coordinates of all true pixels in peaks
    coords = xp.argwhere(peaks)
    coords_3d = coords[:, 1:]
    coords_3d_np = coords_3d.get()
    tree = spatial.cKDTree(coords_3d_np)

    # Query the tree for all points within 1 pixel of each point
    close_points = tree.query_ball_point(coords_3d_np, r=2, workers=-1)

    # Get the number of close points for each point
    num_close_points = [len(x) for x in close_points]

    # Get the indices of points that have only 1 close point
    keep_indices = np.argwhere(np.array(num_close_points) == 1).flatten()

    # Get the coordinates of the points that have only 1 close point
    keep_coords = coords_3d[keep_indices]

    # Draw coordinates on single frame
    coord_image = xp.zeros_like(current_frame)
    coord_image[keep_coords[:, 0], keep_coords[:, 1]] = 1

    expand_ld_mask_for_borders = ndi.binary_dilation(dilated_mask)
    ld_borders = expand_ld_mask_for_borders * ~dilated_mask
    coords_ld_borders = xp.argwhere(ld_borders)

    # Get radius of kept coords 3d by finding distance to closest non-ld mask point
    # Create a KDTree of the coordinates
    tree_ld_centroid = spatial.cKDTree(keep_coords.get())
    tree_ld_borders = spatial.cKDTree(coords_ld_borders.get())
    max_ld_radius_px = 20
    sparse_distance_matrix = tree_ld_centroid.sparse_distance_matrix(
        tree_ld_borders, max_distance=max_ld_radius_px, p=2, output_type='coo_matrix'
    )
    sparse_distance_matrix = sparse_distance_matrix.toarray()
    sparse_distance_matrix[sparse_distance_matrix == 0] = np.inf

    # Get min value for each row of the sparse coo matrix where distance is not 0
    min_dist = sparse_distance_matrix.min(axis=1)

    # Get the LDs
    frame_lds = []
    for ld_num, radius in enumerate(min_dist):
        intensity_at_coord = current_frame[keep_coords[ld_num, 0], keep_coords[ld_num, 1]]
        new_ld = LD(row=ld_frame_info.row, column=ld_frame_info.column, field=ld_frame_info.field,
                    combo_name=ld_frame_info.combo_name, timepoint=ld_frame_info.timepoint,
                    num_nuc=ld_frame_info.num_nuc,
                    ld_radius=radius, ld_intensity=intensity_at_coord, ld_area=np.pi * radius ** 2)
        frame_lds.append(new_ld)

    return frame_lds

def get_grouped_objects_from_phenix(all_files):
    """
    Get grouped objects from phenix
    :param all_files: List of all files
    :return: Dictionary of grouped objects
    """
    row_map = {i: chr(i + 64) for i in range(1, 13)}

    # create a dictionary to hold grouped objects
    grouped_objects = {}

    for file in all_files:
        # extract the filename without extension
        filename = os.path.splitext(os.path.basename(file))[0]

        # parse the filename (e.g. r1c1f1p1-ch1sk1)
        match = re.match(r"r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)sk(\d+)", filename)

        if match:
            row, column, field, plane, laser_channel, timepoint = match.groups()

            # create a key for the group
            key = (row_map[int(row)], int(column), int(field), int(laser_channel))

            # add the file to the group
            if key not in grouped_objects:
                grouped_objects[key] = {}
            subkey = int(timepoint)
            if subkey not in grouped_objects[key]:
                grouped_objects[key][subkey] = []
            grouped_objects[key][subkey].append((int(plane), file))

    # sort the grouped objects by timepoint
    for key in grouped_objects:
        grouped_objects[key] = dict(sorted(grouped_objects[key].items()))

    return grouped_objects


def save_csv(frame_lds, csv_name):
    """
    Save the csv
    :param frame_lds: List of LD objects
    :param csv_name: Name of the csv
    """
    if not os.path.exists(csv_name):
        with open(csv_name, 'w') as f:
            f.write('row,column,field,timepoint,combo_name,'
                    'num_nuc,ld_radius,ld_intensity,ld_area\n')
    with open(csv_name, 'a') as f:
        for single_ld in frame_lds:
            f.write(
                f'{single_ld.row},{single_ld.column},{single_ld.field},{single_ld.timepoint},{single_ld.combo_name},'
                f'{single_ld.num_nuc},{single_ld.ld_radius},{single_ld.ld_intensity},{single_ld.ld_area}\n')


if __name__ == "__main__":
    # e.g.
    folder_names = [
        ("folder_top_dir", "folder_sub_dir"),
        # ("folder_top_dir2", "folder_sub_dir2"),
    ]
    num_folders = len(folder_names)
    for folder_num, folder_name in enumerate(folder_names):
        top_dir = rf'path\to\top_dir\{folder_name[0]}\{folder_name[1]}\Images'
        save_dir = r"path\to\save\directory"
        csv_name = os.path.join(save_dir, f'{dt_str}-{folder_name[0]}-{folder_name[1]}-ld_stats.csv')

        all_files = glob.glob(top_dir + r"\*.tiff")
        all_files.sort()

        grouped_objects = get_grouped_objects_from_phenix(all_files)

        groups_to_run = list(set([key[:-1] for key in grouped_objects.keys()]))

        # This uses GPU for processing
        mempool = xp.get_default_memory_pool()
        pinned_mempool = xp.get_default_pinned_memory_pool()

        ms_set = MultiScaleSettings()
        num_groups = len(groups_to_run)
        dt_start = datetime.datetime.now()
        dt_elapsed = None
        total_num_groups = num_groups * num_folders
        for group_num, run_group in enumerate(groups_to_run):
            if dt_elapsed is not None:
                hours_left = dt_elapsed.total_seconds() / 3600 * total_num_groups
            else:
                hours_left = -1.00
            # Total groups is num_groups * num_folders, so percent complete is (group_num + 1) / (num_groups * num_folders)
            percent_complete = (group_num + 1) / (num_groups * num_folders) * 100
            print(f'{dt_hms}: <<Folder {folder_num}/{num_folders}>> <<Group {group_num + 1}/{num_groups}>> '
                  f'({percent_complete:.2f}%, hours left: {hours_left:.2f})')
            nuc_dict = grouped_objects[run_group + (3,)]
            ld_dict = grouped_objects[run_group + (2,)]
            full_nuc_im = get_max_intensity_timelapse(nuc_dict)
            full_ld_im = get_max_intensity_timelapse(ld_dict)

            for frame_num, frame in enumerate(full_ld_im):
                if (frame_num+1) % 10 == 0:
                    print(f'Running frame {frame_num+1} of {len(full_ld_im)}')
                # Free up GPU memory between images
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
                try:
                    frame_lds = run_frame(frame_num, frame, full_nuc_im[frame_num], ms_set)
                    save_csv(frame_lds, csv_name)
                except Exception as e:
                    print(f'Error on frame {frame_num}: {e}')
                    continue

            dt_end = datetime.datetime.now()
            dt_elapsed = dt_end - dt_start
            dt_start = dt_end
            total_num_groups -= 1
