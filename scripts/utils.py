import numpy as np
import os
import re
import scipy.spatial as sp

PIECE_NAMES = dict(bach='Bach: WTC Prelude in C BWV 846',
                   beethoven='Beethoven: Sonata Op. 27 No. 2 Mv. 1',
                   brahms='Brahms Intermezzo Op. 119 No. 3',
                   liszt='Liszt: Batagelle sans tonalitè',
                   schumann_arabeske_excerpt1='Schumann: Arabeske Op. 18 (Excerpt 1)',
                   schumann_arabeske_excerpt2='Schumann: Arabeske Op. 18 (Excerpt 2)',
                   schumann_kreisleriana_excerpt1='Schumann: Kreisleriana (Excerpt 1)',
                   schumann_kreisleriana_excerpt2='Schumann: Kreisleriana (Excerpt 2)',
                   mozart='Mozart: Sonata K 545 Mv. 2')

SCORE_NAMES = dict(bach='Bach_Praeludium_1_WTC',
                   beethoven='Beethoven_Op27_No2',
                   schumann_kreisleriana_excerpt1='Schumann_Kreisleriana_No3_Cut2',
                   schumann_kreisleriana_excerpt2='Schumann_Kreisleriana_No3_Cut1',
                   schumann_arabeske_excerpt1='Schumann_Arabeske_Op18_Cut2',
                   schumann_arabeske_excerpt2='Schumann_Arabeske_Op18_Cut1',
                   mozart='Mozart_K545_Mv2',
                   liszt='Liszt_Bagatelle',
                   brahms='Brahms_Intermezzo_Op119_No2')


def list_files_deep(dir, full_paths=False, filter_ext=None):
    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(dir, '')):
        if len(filenames) > 0:
            for f in filenames:
                if full_paths:
                    all_files.append(os.path.join(dirpath, f))
                else: all_files.append(f)
    if filter_ext is not None:
        return [f for f in all_files if os.path.splitext(f)[1] in filter_ext]
    else:
        return all_files
    
    
def get_file_from_id(list_of_files, music_id):
    found = [i for i in list_of_files if '/'+str(music_id).zfill(2)+'_' in i]
    if len(found) == 1:
#         print(found[0])
        return found[0]
    elif len(found) > 1:
        print("Non unique: {0} {1}, returning first".format(music_id, found))
        return found[0]
    else:
        return None


def davies_bouldin(X, Y, centroids, q=2., p=2.):
   """
   Davies Bouldin Index

   Compute the Davies Bouldin index, a cluster separation measure

   Reference:
   Davies, D. L and Bouldin, D. W. A Cluster Separation Measure, IEEE
   Transactions on Pattern Analysis and Machine Intelligence, Vol 1 No. 2
   April 1979


   Parameters
   ----------
   X : ndarray
       Array containing the datapoints
   Y : ndarray
       Array that says to which cluster does each datapoint belongs to
   centroids: ndarray
       centroids of the clustering
   q : float (default 2)
       Parameter controlling the distance for the dispersion of the datapoints
   p : float (default 2)
       Parameter controlling the distance for the dispersion of the centroids

   Returns
   -------
   R : float
       Davies-Bouldin index
   """
   n_clusters = len(centroids)
   iq = 1. / q

   # Compute dispersion of the datapoints
   S = np.array([np.mean(np.abs(X[np.where(Y == ci)[0]] - centroids[ci]) ** q) ** iq
                 for ci in range(n_clusters)])

   # Compute distance between the centroids
   M = sp.distance.squareform(sp.distance.pdist(centroids, 'minkowski', p))

   r_idx = np.arange(n_clusters)
   R = []
   for i in range(n_clusters):
       vi = np.where(r_idx != i)[0]
       rij = np.max((S[i] + S[vi]) / M[i, vi])

       R.append(rij)

   return np.mean(R)
