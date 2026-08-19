from config import TrainingConfig


def extend_model_names_with_io_params(
    model_name: str,
    cfg: TrainingConfig,
    include_rigidity: bool,
    use_class_weights: bool,
    io_label_free: bool,
    include_pet: bool,
) -> str:
    model_name += "_IO_"
    model_name += f"lr{cfg.io_lr:.1e}_it{cfg.io_it}"
    model_name += f"_wNCC{cfg.w_io_ncc:.2f}_wDiceCT{cfg.w_io_dice:.2f}"
    model_name += f"_wJac{cfg.w_io_non_diff:.2f}_wSmooth{cfg.w_io_smooth:.2f}"
    model_name += (
        f"_wBoneRigid{cfg.w_io_bone_rigidity if include_rigidity else 0.0:.2f}"
    )
    model_name += f"_wMTV{cfg.w_io_mtv:.2f}_wMTVmean{cfg.w_io_mtv_avg:.2f}_wJactum{cfg.w_io_jacobian_tumor:.2f}_wTLG{cfg.w_io_tlg:.2f}"
    model_name += f"_wMTVcc{cfg.w_io_mtv_cc:.2f}_wMTVavgcc{cfg.w_io_mtv_avg_cc:.2f}_wTLGcc{cfg.w_io_tlg_cc:.2f}"
    print("warning using IO")
    if use_class_weights:
        model_name += "_classweights"
        print("warning using class weights")
    if io_label_free:
        model_name += "_labelfree"
    if include_pet is False:
        model_name += "_noPET"

    return model_name
