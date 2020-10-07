import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import partitura
import pandas as pd

from basismixer.performance_codec import PerformanceCodec, get_performance_codec
from basismixer.utils import notewise_to_onsetwise, onsetwise_to_notewise
import scipy.stats as stats
from scipy.interpolate import interp1d
import partitura.musicanalysis.key_identification as kid


def load_performance(spart_or_xml_fn, match_fn, perf_codec=None, loudness_fn=None):
    """
    Extract performance parameters from an alignment

    Parameters
    ----------
    spart_or_xml_fn : `partitura.score.Part` or filename
        File containg the score in MusicXML, or an already parsed `partitura.score.Part` object.
    match_fn : filename
        Match file corresponding to a performance of the score defined in `spart_or_xml_fn`.
    perf_codec: `PerformanceCodec` or None
        PerformanceCodec instance defining the expressive parameters to extract from the performance.
    loudness_fn: filename or None
        File to the loudness curves corresponding to the score defined in `spart_or_xml_fn`.
    
    Returns
    -------
    expressive_parameters : `pd.DataFrame`
        DataFrame instance containig the performance parameters.
    beat_map : interp1d
        Interpolation function mapping performed time in seconds to score time in beats.
    inv_beat_map: interp1d
        Interpolation function mapping score time in beats to performed time in seconds.
    """
    if isinstance(spart_or_xml_fn, partitura.score.Part):
        spart = spart_or_xml_fn
    else:
        spart = partitura.load_musicxml(spart_or_xml_fn, force_note_ids='keep')

    if perf_codec is None:
        perf_codec = get_performance_codec(['beat_period', 'velocity_trend',
                                            'velocity_dev', 'timing',
                                            'articulation_log'])

    ppart, alignment = partitura.load_match(match_fn)
    score_array = spart.note_array
    perf_array = ppart.note_array

    loudness = None
    if loudness_fn is not None:
        loudness = np.loadtxt(loudness_fn, delimiter=',')


    snote_dict = dict([(n['id'], n) for n in score_array])
    pnote_dict = dict([(n['id'], n) for n in perf_array])
    alignment_onsets = []
    for al in alignment:
        if al['label'] == 'match':

            if al['score_id'] in snote_dict and al['performance_id'] in pnote_dict:
                alignment_onsets.append((snote_dict[al['score_id']]['onset'],
                                         pnote_dict[al['performance_id']]['p_onset']))

    alignment_onsets = np.array(alignment_onsets)
    _expressive_parameters, snote_ids, unique_onset_idxs = perf_codec.encode(spart, ppart, alignment,
                                                                             return_u_onset_idx=True)

    onsets = np.array([snote_dict[sni]['onset'] for sni in snote_ids])
    unique_onsets = np.unique(onsets)
    unique_onsets.sort()
    alignment_onsets = notewise_to_onsetwise(alignment_onsets, unique_onset_idxs)

    beat_map = interp1d(alignment_onsets[:, 1], alignment_onsets[:, 0], kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    inv_beat_map = interp1d(alignment_onsets[:, 0], alignment_onsets[:, 1], kind='linear',
                        bounds_error=False, fill_value='extrapolate')

    expressive_parameters = pd.DataFrame(notewise_to_onsetwise(_expressive_parameters, unique_onset_idxs))
    expressive_parameters['s_onsets'] = unique_onsets
    expressive_parameters['p_onsets'] = inv_beat_map(unique_onsets)
    corr = kid._similarity_with_pitch_profile(spart.note_array)
    majorness = corr.max() if kid.KEYS[corr.argmax()][1] == 'major' else -corr.max()
    expressive_parameters['majorness'] = np.ones(len(unique_onsets)) * majorness

    if loudness is not None:
        lix = 2
        loudness_map = interp1d(beat_map(loudness[:, 0]), loudness[:, 2], kind='linear',
                                bounds_error=False,
                                fill_value=(loudness[0, lix], loudness[-1, lix]))
        expressive_parameters['loudness'] = loudness_map(unique_onsets)
        expressive_parameters['cresc_factor'] = np.ones_like(unique_onsets) * loudness[:, -1].mean()

    del expressive_parameters['timing']
         
    return expressive_parameters, beat_map, inv_beat_map
