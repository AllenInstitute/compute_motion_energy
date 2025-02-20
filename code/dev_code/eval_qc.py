"""File for quality control"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Union
from PIL import Image
import glob
import numpy as np
import pandas as pd
from aind_data_schema.core.quality_control import (QCEvaluation, QCMetric,
                                                   QCStatus, QualityControl,
                                                   Stage, Status)
from aind_data_schema_models.modalities import Modality
from omegaconf import DictConfig
import rerun as rr

PathLike = Union[str, Path]

def eval_pixel_error(value: float, threshold: float = 10):
    """Evaluation metrics pass condition"""
    if value < threshold:
        return status_pass
    elif value > 100:
        return status_pending
    else:
        return status_fail


status_pass = QCStatus(
    status=Status.PASS,
    evaluator="Automated",
    timestamp=datetime.utcnow().isoformat(),
)
status_fail = QCStatus(
    status=Status.FAIL,
    evaluator="Automated",
    timestamp=datetime.utcnow().isoformat(),
)

def summarize_eval_metrics(eval_metrics_dir: PathLike) -> PathLike:
    """Collect evaluation metrics violin plots and save to one pdf file

    Args:
        eval_metrics_dir (PathLike): path to the evaluation metrics plots

    Returns:
        generate a PDF report summarizing the analysis.
    """
    assert os.path.isdir(eval_metrics_dir)

    files_to_collect = []
    temporal_norm_path = os.path.join(
        eval_metrics_dir, "temporal_norm_bodypart_mean_violin.png"
    )
    if not os.path.exists(temporal_norm_path):
        raise FileNotFoundError(f"File not found: {temporal_norm_path}")
    else:
        files_to_collect.append(temporal_norm_path)
        
    pcasingeview_path = os.path.join(
        eval_metrics_dir, "pca_singleview_bodypart_mean_violin.png"
    )
    if os.path.isfile(pcasingeview_path):
        files_to_collect.append(pcasingeview_path)

    confidence_path = os.path.join(
        eval_metrics_dir, "confidence_bodypart_mean_violin.png"
    )
    if not os.path.exists(confidence_path):
        raise FileNotFoundError(f"File not found: {confidence_path}")
    else:
        files_to_collect.append(confidence_path)  
        
    # Open the PNG images
    images = [Image.open(file).convert("RGB") for file in files_to_collect]

    # Save the images as a single PDF
    output_file = os.path.join(eval_metrics_dir, "eval_metrics_summary.pdf")
    images[0].save(output_file, save_all=True, append_images=images[1:])
    
    return output_file

def summarize_example_labeled_frames(eval_metrics_dir: PathLike) -> None:
    """Collect example labeled frames and save to one pdf file

    Args:
        eval_metrics_dir (PathLike): path to the exmaple frames

    Returns:
        generate a PDF report showing example labeled frames.
    """
    def generate_pdf(frames_dir, output_file):
        assert os.path.isdir(frames_dir)
        files_to_collect = glob.glob(os.path.join(frames_dir, "*.png"))
        if not files_to_collect:
            raise FileNotFoundError(f"No example frames found under: {frames_dir}")
        images = [Image.open(file).convert("RGB") for file in files_to_collect]
        images[0].save(output_file, save_all=True, append_images=images[1:])
        

    # collect example frames with high confidence level
    frames_dir = os.path.join(eval_metrics_dir, "highest_confidence", "frames")
    output_file = os.path.join(eval_metrics_dir, "example_frames_with_high_conf.pdf")
    generate_pdf(frames_dir, output_file)

    # collect example frames with low confidence level
    frames_dir = os.path.join(eval_metrics_dir, "lowest_confidence", "frames")
    output_file = os.path.join(eval_metrics_dir, "example_frames_with_low_conf.pdf")
    generate_pdf(frames_dir, output_file)

    
def eval_qc(
    eval_outputs_dir: PathLike, 
    test_video_names: List[str]
) -> None:
    """Perform quality control to summarize evaluation metrics

    Args:
        eval_outputs_dir: path to the evaluation outputs
        test_video_names: list of testing video names

    Returns:
        generate quality_control.json
    """
    print(f"test_video_names: {test_video_names}")

    qc = QualityControl(evaluations=[])

    # ----------------------------------#
    # report eval metrics summary
    # ----------------------------------#
    for video_name in test_video_names:
        QCMetric_list = []
        
        eval_metrics_dir = os.path.join(
            eval_outputs_dir, video_name, "eval_metrics"
        )
        
        pdf_file = summarize_eval_metrics(eval_metrics_dir)
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"File not found: {pdf_file}")
            
        # update the path to the pdf_file
        pdf_file = os.path.join(
            "eval_outputs", 
            video_name, 
            "eval_metrics", 
            "eval_metrics_summary.pdf"
        )

        cur_metric = QCMetric(
            name=f"eval_metrics_summary",
            value="",
            status_history=[status_pass],
            reference=pdf_file,
        )
        QCMetric_list.append(cur_metric)

        cur_evaluation = QCEvaluation(
            name=f"{video_name}_eval_metrics_summary",
            modality=Modality.BEHAVIOR_VIDEOS,
            stage=Stage.PROCESSING,
            metrics=QCMetric_list,
            notes="Evaluation metrics summary report",
        )

        qc.evaluations.append(cur_evaluation)

    # ----------------------------------#
    # report example labeled frames
    # ----------------------------------#
    for video_name in test_video_names:
        QCMetric_list = []

        eval_frames_dir = os.path.join(
            eval_outputs_dir, video_name, "example_frames_labeled"
        )
        
        summarize_example_labeled_frames(eval_frames_dir)

        pdf_file = os.path.join(
            "eval_outputs", 
            video_name, 
            "example_frames_labeled", 
            "example_frames_with_high_conf.pdf"
        )
        cur_metric = QCMetric(
            name=f"{video_name}_example_frames_with_high_conf",
            value="",
            status_history=[status_pass],
            reference=pdf_file,
        )
        QCMetric_list.append(cur_metric)

    
        pdf_file = os.path.join(
            "eval_outputs",
            video_name, 
            "example_frames_labeled", 
            "example_frames_with_low_conf.pdf"
        )
        cur_metric = QCMetric(
            name=f"{video_name}_example_frames_with_low_conf",
            value="",
            status_history=[status_pass],
            reference=pdf_file,
        )
        QCMetric_list.append(cur_metric)

        cur_evaluation = QCEvaluation(
            name=f"{video_name}_example_labeled_frames",
            modality=Modality.BEHAVIOR_VIDEOS,
            stage=Stage.PROCESSING,
            metrics=QCMetric_list,
            notes="Example labeled frames summary report",
        )
        
        qc.evaluations.append(cur_evaluation)

    # ----------------------------------#
    # qc on temporal_norm_bodypart_mean metrics
    # show the trajectories of each bodypart
    # ----------------------------------#
    for video_name in test_video_names:
        QCMetric_list = []

        eval_metrics_dir = os.path.join(
            eval_outputs_dir, video_name, "eval_metrics"
        )

        file_path = os.path.join(eval_metrics_dir, "temporal_norm_bodypart_mean.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        trajectory_dir = os.path.join(
            "eval_outputs", video_name, "trajectory"
        ) # note: provide relative path
        
        for bodypart in df.columns[1:-1]:
            mean = float(df[bodypart].values[0].split("±")[0])
            threshold = 20  # TODO
            print(f"{bodypart} temporal_norm error: {mean} pixels")

            cur_metric = QCMetric(
                name=bodypart,
                value=mean,
                status_history=[eval_pixel_error(value=mean, threshold=threshold)],
                description=f"Pass when lower than {threshold}",
                reference=os.path.join(trajectory_dir, f"{bodypart}_TimeSeries.png"),
            )
            QCMetric_list.append(cur_metric)

        evaluation_temporal_norm = QCEvaluation(
            name=f"{video_name}_Temporal_norm",
            modality=Modality.BEHAVIOR_VIDEOS,
            stage=Stage.PROCESSING,
            metrics=QCMetric_list,
            notes="Pass when Temporal_norm lower than predefined threhold",
        )
        print(
            f"**evaluation_temporal_norm.status for {video_name}**:",
            evaluation_temporal_norm.status,
        )
        qc.evaluations.append(evaluation_temporal_norm)

    # ----------------------------------#
    # show rerun
    # ----------------------------------#
    for video_name in test_video_names:
        QCMetric_list = []
        rrd_file = os.path.join(
            "eval_outputs",
            video_name,
            f"rerun_v{rr.__version__}.rrd", # TODO: add more rrd files
        )
        labeled_video_metric = QCMetric(
            name=f"vis_pose_tracking",
            value="",
            status_history=[status_pass],
            reference=rrd_file,
        )
        QCMetric_list.append(labeled_video_metric)

        evaluation_rerun = QCEvaluation(
            name=f"{video_name}_rerun",
            modality=Modality.BEHAVIOR_VIDEOS,
            stage=Stage.PROCESSING,
            metrics=QCMetric_list,
            notes="Visualize labeled video and trajectories",
        )

        qc.evaluations.append(evaluation_rerun)


    print(qc.status)
    qc.write_standard_file(output_directory=eval_outputs_dir)
