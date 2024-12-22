import cv2
import numpy as np

from pyCFTrackers.lib.utils import get_img_list, get_ground_truthes, APCE, PSR
from pyCFTrackers.cftracker.mosse import MOSSE
from pyCFTrackers.cftracker.csk import CSK
from pyCFTrackers.cftracker.kcf import KCF
from pyCFTrackers.cftracker.cn import CN
from pyCFTrackers.cftracker.dsst import DSST
from pyCFTrackers.cftracker.staple import Staple
from pyCFTrackers.cftracker.dat import DAT
from pyCFTrackers.cftracker.eco import ECO
from pyCFTrackers.cftracker.bacf import BACF
from pyCFTrackers.cftracker.csrdcf import CSRDCF
from pyCFTrackers.cftracker.samf import SAMF
from pyCFTrackers.cftracker.ldes import LDES
from pyCFTrackers.cftracker.mkcfup import MKCFup
from pyCFTrackers.cftracker.strcf import STRCF
from pyCFTrackers.cftracker.mccth_staple import MCCTHStaple
from pyCFTrackers.lib.eco.config import otb_deep_config, otb_hc_config
from pyCFTrackers.cftracker.config import staple_config, ldes_config, dsst_config, csrdcf_config, mkcf_up_config, mccth_staple_config


all_trackers = ['MOSSE',
                'CSK',
                'CN',
                'DSST',
                'Staple',
                'Staple-CA',
                'KCF_CN',
                'KCF_GRAY',
                'KCF_HOG',
                'DCF_GRAY',
                'DCF_HOG',
                'DAT',
                'ECO-HC',
                'ECO',
                'BACF',
                'CSRDCF',
                'CSRDCF-LP',
                'SAMF',
                'LDES',
                'DSST-LP',
                'MKCFup',
                'MKCFup-LP',
                'STRCF',
                'MCCTH-Staple',
                'MCCTH']


class PyTracker:
    def __init__(self, tracker_type):
        self.tracker_type = tracker_type
        print(f"create tracker named {self.tracker_type}")
        if self.tracker_type == 'MOSSE':
            self.tracker = MOSSE()
        elif self.tracker_type == 'CSK':
            self.tracker = CSK()
        elif self.tracker_type == 'CN':
            self.tracker = CN()
        elif self.tracker_type == 'DSST':
            self.tracker = DSST(dsst_config.DSSTConfig())
        elif self.tracker_type == 'STAPLE':
            self.tracker = Staple(config=staple_config.StapleConfig())
        elif self.tracker_type == 'STAPLE-CA':
            self.tracker = Staple(config=staple_config.StapleCAConfig())
        elif self.tracker_type == 'KCF_CN':
            self.tracker = KCF(features='cn', kernel='gaussian')
        elif self.tracker_type == 'KCF_GRAY':
            self.tracker = KCF(features='gray', kernel='gaussian')
        elif self.tracker_type == 'KCF_HOG':
            self.tracker = KCF(features='hog', kernel='gaussian')
        elif self.tracker_type == 'DCF_GRAY':
            self.tracker = KCF(features='gray', kernel='linear')
        elif self.tracker_type == 'DCF_HOG':
            self.tracker = KCF(features='hog', kernel='linear')
        elif self.tracker_type == 'DAT':
            self.tracker = DAT()
        elif self.tracker_type == 'ECO-HC':
            self.tracker = ECO(config=otb_hc_config.OTBHCConfig())
        elif self.tracker_type == 'ECO':
            self.tracker = ECO(config=otb_deep_config.OTBDeepConfig())
        elif self.tracker_type == 'BACF':
            self.tracker = BACF()
        elif self.tracker_type == 'CSRDCF':
            self.tracker = CSRDCF(config=csrdcf_config.CSRDCFConfig())
        elif self.tracker_type == 'CSRDCF-LP':
            self.tracker = CSRDCF(config=csrdcf_config.CSRDCFLPConfig())
        elif self.tracker_type == 'SAMF':
            self.tracker = SAMF()
        elif self.tracker_type == 'LDES':
            self.tracker = LDES(ldes_config.LDESDemoLinearConfig())
        elif self.tracker_type == 'DSST-LP':
            self.tracker = DSST(dsst_config.DSSTLPConfig())
        elif self.tracker_type == 'MKCFup':
            self.tracker = MKCFup(config=mkcf_up_config.MKCFupConfig())
        elif self.tracker_type == 'MKCFup-LP':
            self.tracker = MKCFup(config=mkcf_up_config.MKCFupLPConfig())
        elif self.tracker_type == 'STRCF':
            self.tracker = STRCF()
        elif self.tracker_type == 'MCCTH-Staple':
            self.tracker = MCCTHStaple(
                config=mccth_staple_config.MCCTHOTBConfig())
        elif self.tracker_type == 'MCCTH':
            self.tracker = MCCTH(config=mccth_config.MCCTHConfig())
        else:
            raise NotImplementedError

    def init(self, img, box):
        """
        Args:
            img ():
            box ():  xywh
        """
        self.tracker.init(img, box)

    def track(self, img,  verbose=True):
        """

        Args:
            img ():
            verbose ():

        Returns:  xywh box

        """
        bbox = self.tracker.update(img, vis=verbose)
        return bbox

    def drawVerbose(self, img, bbox):
        """

        Args:
            img (): image
            bbox (): xywh box
        """
        x1, y1, w, h = bbox
        if len(img.shape) == 2:  # gray image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        height, width = img.shape[:2]
        score = self.tracker.score
        apce = APCE(score)
        psr = PSR(score)
        F_max = np.max(score)
        size = self.tracker.crop_size
        # map score to [0, 1] to colormap
        score = cv2.resize(score, size)
        score -= score.min()
        score = score / score.max()
        score = (score * 255).astype(np.uint8)
        # score = 255 - score
        score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
        center = (int(x1+w/2), int(y1+h/2))
        x0, y0 = center
        x0 = np.clip(x0, 0, width-1)
        y0 = np.clip(y0, 0, height-1)
        center = (x0, y0)
        xmin = int(center[0]) - size[0] // 2
        xmax = int(center[0]) + size[0] // 2 + size[0] % 2
        ymin = int(center[1]) - size[1] // 2
        ymax = int(center[1]) + size[1] // 2 + size[1] % 2
        left = abs(xmin) if xmin < 0 else 0
        xmin = 0 if xmin < 0 else xmin
        right = width - xmax
        xmax = width if right < 0 else xmax
        right = size[0] + right if right < 0 else size[0]
        top = abs(ymin) if ymin < 0 else 0
        ymin = 0 if ymin < 0 else ymin
        down = height - ymax
        ymax = height if down < 0 else ymax
        down = size[1] + down if down < 0 else size[1]
        score = score[top:down, left:right]
        crop_img = img[ymin:ymax, xmin:xmax]
        score_map = cv2.addWeighted(crop_img, 0.6, score, 0.4, 0)
        img[ymin:ymax, xmin:xmax] = score_map
        show_img = cv2.rectangle(img, (int(x1), int(
            y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 1)
        """
            cv2.putText(show_frame, 'APCE:' + str(apce)[:5], (0, 250), cv2.FONT_HERSHEY_COMPLEX, 2,
                        (0, 0, 255), 5)
            cv2.putText(show_frame, 'PSR:' + str(psr)[:5], (0, 300), cv2.FONT_HERSHEY_COMPLEX, 2,
                        (255, 0, 0), 5)
            cv2.putText(show_frame, 'Fmax:' + str(F_max)[:5], (0, 350), cv2.FONT_HERSHEY_COMPLEX, 2,
                        (255, 0, 0), 5)
            """
        return show_img
