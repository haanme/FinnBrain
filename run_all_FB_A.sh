#!/bin/sh

#python preprocess_mergedDTI.py --dicomdir "/Users/eija/Documents/FinnBrain/FB_13_2014spring/FB028043" --DTI1 ep2d_diff_tensor_34_pat2_27 --DTI2 ep2d_diff_tensor_35_pat2_33 --DTI3 ep2d_diff_tensor_36_pat2_19 --T1 t1_mpr_sag_p2_iso_10_26 --subject FB028043 --FieldMap_Mag gre_field_mapping_whisper_5 --FieldMap_Pha gre_field_mapping_whisper_6
python dicom2streamlines_phase3_from_john3.py --dicomdir "/Users/eija/Documents/FinnBrain/FB_13_2014spring/FB032153" --DTI1 ep2d_diff_tensor_34_pat2_27 --DTI2 ep2d_diff_tensor_35_pat2_33 --DTI3 ep2d_diff_tensor_36_pat2_19 --T1 t1_mpr_sag_p2_iso_10_26 --subject from_john3 --FieldMap_Mag gre_field_mapping_whisper_5 --FieldMap_Pha gre_field_mapping_whisper_6
python preprocess_mergedDTI.py --dicomdir "/Users/eija/Documents/FinnBrain/FB_13_2014spring/FB032153" --DTI1 ep2d_diff_tensor_34_pat2_27 --DTI2 ep2d_diff_tensor_35_pat2_33 --DTI3 ep2d_diff_tensor_36_pat2_19 --T1 t1_mpr_sag_p2_iso_10_26 --subject from_john3 --FieldMap_Mag gre_field_mapping_whisper_5 --FieldMap_Pha gre_field_mapping_whisper_6

