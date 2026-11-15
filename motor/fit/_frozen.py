"""Main's calibration table, frozen (origin/main 227b29b2's `calibration.TABLE`, R_cal per bone group as a quaternion
x, y, z, w): the stage's record judges each group's turn on it beside the tree's own table, so a change of the table
cannot make the solver's turns look better than they are (#70, the audit's item 2). Never edited to follow the tree."""
from scipy.spatial.transform import Rotation

TABLE = {
    "Hips": (0.663583, 0.748103, 0.000268687, -0.000290935),
    "Spine1": (0.756786, 0.653663, 0.000257497, -0.00029812),
    "Spine2": (0.596189, 0.802841, 0.00165502, 0.00162641),
    "Chest": (0.683552, 0.729902, 0.000118136, -0.000455094),
    "Neck1": (0.770799, 0.637077, -0.000728835, -0.00110972),
    "Neck2": (0.770799, 0.637077, -0.000728836, -0.00110972),
    "Head": (0.7035, 0.710695, 0.000156141, -0.000405753),
    "LeftShoulder": (-0.446904, -0.498353, 0.576873, 0.468122),
    "RightShoulder": (0.498428, -0.446843, -0.46818, 0.576809),
    "LeftArm": (-0.00317606, 0.0327142, 0.698884, 0.714479),
    "RightArm": (0.0327143, 0.00317603, 0.714479, -0.698884),
    "LeftForeArm": (-0.0787067, -0.0462542, 0.679867, 0.727631),
    "RightForeArm": (-0.0462542, 0.0787067, 0.727631, -0.679867),
    "LeftHand": (0.0222354, -0.198252, 0.673792, 0.711482),
    "RightHand": (-0.198308, -0.0221821, 0.711448, -0.673813),
    "LeftLeg": (-0.707106, 0.707106, -0.00110608, 0.00110614),
    "RightLeg": (0.707106, 0.707106, 0.00110614, 0.00110608),
    "LeftShin": (-0.698215, 0.715888, 0.000634178, 0.000618516),
    "RightShin": (0.715888, 0.698215, 0.00061852, -0.000634175),
    "LeftFoot": (0.985798, -0.16093, -0.0379763, -0.0293416),
    "RightFoot": (0.16092, 0.985791, 0.0295935, -0.0380175),
    "LeftToeBase": (0.994273, -0.030612, 0.0988721, -0.0266242),
    "RightToeBase": (0.0306151, 0.994272, 0.026656, 0.0988718),
}


def frozen_turn(group):
    """R_cal of `group` (3 x 3) in main's frozen table."""
    return Rotation.from_quat(TABLE[group]).as_matrix()
