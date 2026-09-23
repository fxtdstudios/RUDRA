"""Fetch the RUDRA corpus against docs/CORPUS_ACQUISITION.md.

Runs on the Windows box, standard library only -- no pip install:

    python training\fetch_corpus.py --plan
    python training\fetch_corpus.py --check
    python training\fetch_corpus.py --fetch --only netflix_sol_levante

Four modes, in the order they are meant to be used.

  --plan    Resolve every source to a concrete file list and print the byte
            total. Lists the S3 buckets and queries the Poly Haven API, writes
            nothing, downloads nothing.
  --check   HEAD every resolved URL. Reports status and whether the server's
            Content-Length matches what the plan expected. Run this before
            moving 250 GB: a dead URL costs seconds here and an hour there.
  --fetch   Download, resuming part files by HTTP Range, hashing as it writes.
  --report  Re-read the manifests already on disk and print the corpus state.

WHY THE URLS ARE NOT ALL HARDCODED

Three of the sources publish a machine-readable index -- S3 ListObjectsV2 for
the Netflix and Laval buckets, a JSON API for Poly Haven. Those are discovered
at run time, because a hardcoded key list goes stale silently and a listing
cannot. Only the sets with no index carry literal URLs, and --check is what
proves those are still alive.

WHAT THIS WILL NOT DO

Sources whose access is 'gated' or 'paid' are never fetched. They need a person
to fill in a form, accept a EULA, or authorise a purchase. --plan prints them as
a to-do list with the exact URL or address. Sources whose licence text forbids
training are listed under 'refused' with the sentence that refuses, and are not
fetched even with --only.

EVERY SET GETS A licence.txt

Written next to the data, verbatim, with the URL it was read from and the date.
A corpus whose terms are not written down cannot later be shown to permit
anything -- and 'commercial_ok' is recorded per set, not inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

UA = "rudra-corpus-fetch/1.0 (FXTD Studios; SDR-to-HDR research)"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
CHUNK = 4 * 1024 * 1024


# --------------------------------------------------------------------------
# source table
# --------------------------------------------------------------------------

@dataclass
class Source:
    sid: str
    name: str
    classes: str                 # which corpus classes it supplies
    scenes: str                  # independent scenes, the unit that matters
    licence: str
    licence_url: str
    licence_quote: str           # verbatim, from the source page
    commercial_ok: str           # yes | no | unclear
    access: str                  # direct | gated | paid | refused
    note: str = ""
    # direct only:
    kind: str = ""               # s3 | static | polyhaven
    base_url: str = ""           # full listing base, trailing slash
    prefixes: tuple = ()
    include: str = ""            # regex an S3 key must match
    urls: tuple = ()
    resolution: str = ""         # polyhaven
    default: bool = True         # fetched when --only is not given
    # gated / paid / refused:
    action: str = ""


SOURCES: list[Source] = [

    # ---- tier 1: direct, licence clean, commercial granted ---------------

    Source(
        sid="netflix_sol_levante",
        name="Netflix Open Content -- Sol Levante",
        classes="native HDR/SDR pair; screens and graphic emissive",
        scenes="1 title (~4 min)",
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        licence_quote=(
            "Our open source content is available under the Creative Commons "
            "Attribution 4.0 International Public License."
        ),
        commercial_ok="yes",
        access="direct",
        note=(
            "The ONLY Netflix Open Content title with an SDR master. Chimera, "
            "Meridian, Cosmos Laundromat, Sparks and Nocturne are HDR-only, so "
            "this is the only native pair in the bucket. ~53 GB for the pair; "
            "the 155 GB VDM TGA set is excluded by the include pattern."
        ),
        kind="s3",
        base_url="https://s3.amazonaws.com/download.opencontent.netflix.com/",
        prefixes=("hdr10/", "sdr/"),
        include=r"SolLevante.*\.(mov|mxf)$",
    ),

    Source(
        sid="netflix_sparks",
        name="Netflix Open Content -- Sparks (4000 nit)",
        classes="specular exteriors; practical lights; water/chrome",
        scenes="1 title, multi-shot",
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        licence_quote=(
            "Our open source content is available under the Creative Commons "
            "Attribution 4.0 International Public License."
        ),
        commercial_ok="yes",
        access="direct",
        note=(
            "The highest peak luminance found anywhere with unambiguous "
            "commercial rights: 4000-nit P3/PQ. Our corpus median is 1007 nits. "
            "The 4000-nit IMF zip is 46.3 GB; there is also a 1000-nit HDR10 "
            "TIFF sequence. --plan prints everything the prefix holds so the "
            "4000-nit master is picked deliberately, not guessed."
        ),
        kind="s3",
        base_url="https://s3.amazonaws.com/download.opencontent.netflix.com/",
        prefixes=("sparks/",),
        include=r"(4000nits|ACES|EXR).*\.(zip|exr|tif|mov|mxf)$",
    ),

    Source(
        sid="liu_hdrv",
        name="Linkoping HDRv",
        classes="specular exteriors; night interiors; water",
        scenes="12",
        licence="CC BY-SA 4.0",
        licence_url="https://creativecommons.org/licenses/by-sa/4.0/",
        licence_quote=(
            "All data, code and other information in the HDRv repository may be "
            "used freely under the terms of the creative commons license "
            "CC BY-SA 4.0."
        ),
        commercial_ok="yes",
        access="direct",
        note=(
            "12 independent scenes, OpenEXR 1280x720, capture rig rated over 24 "
            "f-stops. Small (11 GB total) and the best scene-count-per-gigabyte "
            "in the survey. ShareAlike is viral -- check how it interacts with "
            "distributed weights before this one lands in a shipped checkpoint. "
            "hdrv.org itself is robots-blocked; this Linkoping mirror is the "
            "working host."
        ),
        kind="static",
        urls=tuple(
            "https://computergraphics.on.liu.se/hdrv_itn_liu/clips/" + n
            for n in (
                "Astronauts.zip", "window.zip", "students.zip", "hallway.zip",
                "hallway2.zip", "water.zip", "bridge2.zip", "bridge.zip",
                "exhibition_area.zip", "river.zip", "C.zip", "kaken.zip",
            )
        ),
    ),

    Source(
        sid="pandora",
        name="PanDORA (Laval)",
        classes="indoor practical lights; absolute-ish bracketed HDR",
        scenes="14 scenes / 195 panoramas",
        licence="custom, permissive",
        licence_url="https://lvsn.github.io/pandora/dataset/index.html",
        licence_quote=(
            "You are free to: Use, process, and build upon this dataset for any "
            "purpose, including academic research, education, and commercial "
            "research, provided that you cite the original paper. You may not: "
            "Redistribute, re-host, mirror, or otherwise make this dataset "
            "publicly available from a source other than the official project "
            "page."
        ),
        commercial_ok="yes",
        access="direct",
        note=(
            "The only Laval set that is not EULA-gated. 'Any purpose' plus "
            "'build upon' covers training; the bar is on redistributing the "
            "DATA, which does not bar weights. HDR EXRs land in "
            "<scene>/GT/GT_exr/ after extraction."
        ),
        kind="s3",
        base_url="https://hdrdb-public.s3.valeria.science/",
        prefixes=("pandora/",),
        include=r"\.zip$",
    ),

    Source(
        sid="polyhaven",
        name="Poly Haven HDRIs (16k EXR)",
        classes="renderer path, not photographic",
        scenes="~960 panoramas",
        licence="CC0 1.0",
        licence_url="https://polyhaven.com/license",
        licence_quote=(
            "We release all our assets under the CC0 license...which allows you "
            "to do whatever you want with them, including training AI models."
        ),
        commercial_ok="yes",
        access="direct",
        default=False,
        note=(
            "OFF BY DEFAULT and that is deliberate. We already hold ~963 of "
            "these, and 205 of them decode to between 1e6 and 1.33e38 nits -- "
            "they are the ONLY clipped pixels in the entire training corpus and "
            "every one of them is wrong. Re-downloading before the decode is "
            "fixed re-imports the bug. Fix pipeline/hdr_io.py first, re-measure, "
            "then --only polyhaven if the fix needs pristine files. The API "
            "reports md5 per file, which this script verifies."
        ),
        kind="polyhaven",
        resolution="16k",
    ),

    # ---- tier 2: direct, purpose-restricted licence ----------------------

    Source(
        sid="aswf_stem2",
        name="ASC StEM2 'The Mission' (ASWF DPEL)",
        classes="native HDR/SDR pair; day/night interior and exterior",
        scenes="18 580 frames, one 17-min film; shot count unpublished",
        licence="ASWF Digital Assets License v1.1",
        licence_url="https://dpel.aswf.io/asc-stem2/",
        licence_quote=(
            "Redistribution and use of these digital assets, with or without "
            "modification, solely for education, training, research, software "
            "and hardware development, performance benchmarking (including "
            "publication of benchmark results and permitting reproducibility of "
            "the benchmark results by third parties), or software and hardware "
            "product demonstrations, are permitted provided that the following "
            "conditions are met: [...] 2. Publications showing images derived "
            "from these digital assets must include the above copyright notice."
        ),
        commercial_ok="unclear",
        access="direct",
        default=False,
        note=(
            "Matched SDR-100-nit Rec.709 and HDR-1000-nit Rec.2020 PQ release "
            "masters from one ACES pipeline -- a real native pair, not a "
            "synthetic tone-map. 'Solely' gates use to a list: 'research' and "
            "'training' plausibly cover us, SHIPPING WEIGHTS IS NOT ADDRESSED. "
            "Fetched with commercial_ok=unclear; ask ASWF in writing before it "
            "becomes load-bearing. Off by default because the EXR set is 1.4 TB "
            "-- the two ProRes masters alone are 190 GB and are enough to start. "
            "URLs are literal here and unverified from my side: run --check."
        ),
        kind="static",
        urls=(
            "https://aswf-dpel-assets.s3.amazonaws.com/asc-stem2/"
            "ASC_StEM2_178_UHD_24_100nits_Rec709_Stereo_ProRes422HQ.mov",
            "https://aswf-dpel-assets.s3.amazonaws.com/asc-stem2/"
            "ASC_StEM2_178_UHD_ST2084_1000nits_Rec2020_Stereo_ProRes4444XQ.mov",
        ),
    ),

    # ---- gated: a person has to do this ---------------------------------

    Source(
        sid="live_tmhdr",
        name="LIVE-TMHDR (UT Austin)",
        classes="native HDR/SDR pair, colourist-graded",
        scenes="40",
        licence="LIVE permissive",
        licence_url="https://live.ece.utexas.edu/research/LIVE_TMHDR/index.html",
        licence_quote=(
            "Permission is hereby granted, without written agreement and without "
            "license or royalty fees, to use, copy, modify, and distribute this "
            "database (the videos, the results and the source files) and its "
            "documentation for any purpose, provided that the copyright notice "
            "in its entirety appear in all copies of this database, and the "
            "original source of this database, Laboratory for Image and Video "
            "Engineering (LIVE) at the University of Texas at Austin, is "
            "acknowledged in any publication that reports research using this "
            "database."
        ),
        commercial_ok="yes",
        access="gated",
        note=(
            "HIGHEST VALUE PER MINUTE SPENT IN THIS FILE. 40 scenes a "
            "commissioned professional colourist graded by hand -- 'for any "
            "purpose', royalty-free. This one source oversupplies the corpus "
            "class that breaks the synthetic-degradation circularity in the "
            "paper. It is a Google Form."
        ),
        action=(
            "Fill in https://docs.google.com/forms/d/e/"
            "1FAIpQLSd_qiHglRkpMWVxmheHs1cZ-T1iwJiMaqwYHNwZ3n3W9VxHpw/viewform "
            "then drop the download links into --extra-urls, or fetch by hand "
            "into G:\\datasets_rudra\\live_tmhdr\\"
        ),
    ),

    Source(
        sid="live_hdr_vqa",
        name="LIVE HDR VQA + HDRSDR-VQA (UT Austin)",
        classes="specular exteriors; sports; native pair (partial)",
        scenes="31 + 31",
        licence="LIVE permissive",
        licence_url="https://live.ece.utexas.edu/research/LIVEHDR/LIVEHDR_index.html",
        licence_quote=(
            "Permission is hereby granted, without written agreement and without "
            "license or royalty fees, to use, copy, modify, and distribute this "
            "database [...] for any purpose."
        ),
        commercial_ok="yes",
        access="gated",
        note=(
            "Caveat worth knowing before you spend the time: the genuinely "
            "colourist-graded HDRSDR-VQA pairs (Amazon Studios) are NOT "
            "released, only their JOD scores. The 31 downloadable sources use "
            "public NBCU LUTs -- a fixed automatic conversion, i.e. the same "
            "circularity we are trying to escape. Take them for scene count and "
            "dynamic range, not as native pairs."
        ),
        action="https://forms.gle/pAHApGPdydjQ71W37",
    ),

    Source(
        sid="xdr",
        name="xDR (imec / Ghent + ESPOL)",
        classes="native HDR/SDR pair, one expert graded both sides",
        scenes="10",
        licence="NONE STATED ANYWHERE",
        licence_url="http://telin.ugent.be/~gluzardo/hdr-sdr-dataset/",
        licence_quote=(
            "(no licence statement exists on the landing page, in the paper, or "
            "in the UGent record) -- the grading claim is: 'The same expert "
            "graded both the SDR and HDR video sequences in two separate "
            "pipelines.'"
        ),
        commercial_ok="unclear",
        access="gated",
        note=(
            "Purpose-built for exactly our problem: evaluating inverse tone "
            "mapping, natively graded, 0.03-6000 cd/m2, 17.5 stops, one expert "
            "on both sides. The public portal carries a reduced set at ONE FRAME "
            "PER SECOND, useless for temporal work; the full set is by email. "
            "There is no licence at all, so ask for terms in the same email -- "
            "do not fetch first and ask later."
        ),
        action="Email Gonzalo.Luzardo@imec.be for the full set AND written terms",
    ),

    Source(
        sid="laval_photometric",
        name="Laval Photometric Indoor HDR",
        classes="absolute-luminance reference",
        scenes="2 362 panoramas",
        licence="Laval EULA, two tiers",
        licence_url="http://hdrdb.com/indoor-hdr-photometric/",
        licence_quote=(
            "commercial tier: 'use of the data to create or improve models and "
            "resulting output, with the right to make the output available to "
            "third parties' -- non-profit tier: 'This license does not grant the "
            "right to use this dataset or any derivation of it for commercial "
            "activities.'"
        ),
        commercial_ok="yes, on the paid tier only",
        access="gated",
        note=(
            "The only data in the whole survey that is ABSOLUTELY CALIBRATED in "
            "cd/m2 -- Konica Minolta CL-200A chroma meter, per-channel "
            "regression R2 > 0.985, 3884x7768, 22 f-stops. Given that 205 of our "
            "panoramas currently decode to 1e38 nits, a metered ground truth is "
            "worth the paperwork on its own as a calibration check. And the "
            "commercial tier is the ONLY licence found anywhere that names the "
            "weights question instead of leaving it silent. Ask for that tier "
            "explicitly."
        ),
        action=(
            "Email jflalonde@gel.ulaval.ca -- one EULA covers Photometric "
            "Indoor, Indoor, Outdoor and Face+Lighting. Say commercial tier."
        ),
    ),

    Source(
        sid="sony_venice2",
        name="Sony VENICE 2 8K X-OCN (Copenhagen)",
        classes="specular exteriors; water; architecture; lowlight",
        scenes="9",
        licence="none stated",
        licence_url="https://pro.sony/en_BA/cinematography/cinematography-tips/x-ocn-8k-downloads",
        licence_quote="(no terms of use exposed on the page)",
        commercial_ok="unclear",
        access="gated",
        note="X-OCN LT/ST/XT plus 4K ProRes 4444, 8.6K, Cooke/i metadata.",
        action="Sony Ci registration on the downloads page",
    ),

    Source(
        sid="arri_samples",
        name="ARRI camera sample footage",
        classes="specular exteriors; night interiors; skin; FIRE; water",
        scenes="~20+ locations; 'Encounters' alone is 17 clips",
        licence="NONE -- copyright notice only",
        licence_url="https://www.arri.com/en/learn-help/learn-help-camera-system/camera-sample-footage-reference-image",
        licence_quote=(
            "Please note that this sample footage was created for workflow "
            "evaluation purposes. -- Copyright (c) 2026 Arnold & Richter Cine "
            "Technik GmbH & Co. Betriebs KG. All rights reserved."
        ),
        commercial_ok="unclear",
        access="gated",
        note=(
            "Best free technical match in the survey: ARRIRAW, ARRICORE, LogC3 "
            "and LogC4/AWG4, scene-linear ACES AP0, up to 6.5K, and the "
            "'Encounters' set covers five of our eight classes in one package "
            "INCLUDING the campfire -- class 4 is otherwise almost empty. "
            "Silence is not a grant, so this is a written-request item, not a "
            "download item. FXTD is an ARRI house; the request is likely to "
            "succeed and costs a day."
        ),
        action="Written permission request to digitalworkflow@arri.de",
    ),

    Source(
        sid="dvb_hdr",
        name="DVB / EBU HDR test content",
        classes="specular exteriors; skin",
        scenes="~2-4",
        licence="CC BY 4.0",
        licence_url="https://dvb.org/specifications/verification-validation/hdr-test-content/",
        licence_quote=(
            "This material has been recorded by the EBU, who owns the copyright. "
            "The EBU (licensor) has licensed it under the Creative Commons "
            "Attribution 4.0 license. In short, this means you can freely reuse "
            "and distribute this content, also commercially, for as long you "
            "provide a proper attribution."
        ),
        commercial_ok="yes",
        access="gated",
        note=(
            "Licence is clean but the page exposes no direct URLs to a fetcher "
            "-- the 14 transport streams are click-through links. Low priority "
            "anyway: shot on a Sony Z280 broadcast camcorder, roughly 12-13 "
            "stops, so it will not lift the median or serve the >=14-stop "
            "classes. ~3 GB total if you want it."
        ),
        action="Click the 14 links on the page, or email dvb@dvb.org for originals",
    ),

    # ---- paid ------------------------------------------------------------

    Source(
        sid="hdm_commercial",
        name="HdM-HDR-2014 commercial licence",
        classes="fire; night interiors; skin; welding",
        scenes="~10-11 real setups (16 clips, but re-angles)",
        licence="HdM custom, paid",
        licence_url="https://hdm-stuttgart.de/vmlab/hdm-hdr-2014/",
        licence_quote=(
            "Academic and educational use of the HdM-HDR-2014 data set is free. "
            "We license the commercial use of this HDR-Video data set for a "
            "small contribution towards our production costs. This license "
            "includes the right to show the image sequences on trade shows, "
            "internal pipeline development, and the evaluation of monitors and "
            "tonemapping operators. Please note the redistribution is not "
            "allowed."
        ),
        commercial_ok="paid, and the grant does not mention training",
        access="paid",
        note=(
            "We ALREADY HOLD this data -- it is 46% of the current corpus -- "
            "under the free academic licence. Read the granted rights again: "
            "trade shows, internal pipeline development, monitor and TMO "
            "evaluation. TRAINING IS NOT IN THAT LIST, on either tier. This is "
            "the largest licence exposure in the corpus and it is already in the "
            "pool. Worth a written clarification whatever else happens. Also "
            "note the 16 clips are ~10 setups: Fishing x2, Cars x3, Poker x2, "
            "Showgirl x2 are re-angles, and a manifest counting 16 is wrong."
        ),
        action="Quote and training clarification from Prof. Stefan Grandinetti, HdM Stuttgart",
    ),

    Source(
        sid="ebu_eac2018",
        name="EBU EAC2018 2160p/100 HLG",
        classes="specular exteriors; stadium day/twilight/night",
        scenes="29 clips, ONE venue -- 2-5 real scenes",
        licence="EBU buyer agreement",
        licence_url="https://tech.ebu.ch/publications/eac-2018---test-sequences-2160p100-hlg",
        licence_quote=(
            "The Buyer agrees to notify the EBU should it intend to use the "
            "Content in the presence [of third parties] [...] Buyer shall not "
            "create or establish any implied or direct endorsement of any "
            "product or service."
        ),
        commercial_ok="unclear",
        access="paid",
        note=(
            "EUR 300 member / 1 900 university / 3 800 other, shipped on a "
            "physical SSD. Uncompressed DPX 2160p100 BT.2100 HLG. Poor value "
            "per euro for us: 29 clips of one athletics stadium is a handful of "
            "independent scenes, and independent scenes are the unit that "
            "matters. Listed for completeness, not recommended."
        ),
        action="testsequences@ebu.ch",
    ),

    # ---- refused: the text forbids it ------------------------------------

    Source(
        sid="sjtu_hdr",
        name="SJTU HDR Video Sequences",
        classes="fire (Bonfire); night street; traffic -- technically ideal",
        scenes="16",
        licence="SJTU academic terms",
        licence_url="https://medialab.sjtu.edu.cn/post/sjtu-hdr-video-sequences/",
        licence_quote=(
            "Our dataset is available for free and academic research only, no "
            "commercial use."
        ),
        commercial_ok="no",
        access="refused",
        note=(
            "The most painful exclusion in the file. 16 scenes, 208 GB, "
            "OpenEXR half-float from Sony F65/F55 RAW, over 14 stops -- and it "
            "contains Bonfire, which is the class we are shortest of. Directly "
            "downloadable, no gate. The words 'no commercial use' are explicit, "
            "so this is a fact about the licence and not a judgement about "
            "commerce. Your call as the licensee; say the word and it moves out "
            "of refused with commercial_ok=no recorded on every scene."
        ),
    ),

    Source(
        sid="fraunhofer_8k",
        name="Fraunhofer HHI 8K Berlin",
        classes="specular exteriors; architecture; matched SDR+PQ pairs",
        scenes="7",
        licence="CC BY-NC-ND 4.0",
        licence_url="https://creativecommons.org/licenses/by-nc-nd/4.0/",
        licence_quote=(
            "The video sequences are licensed under the Creative Commons "
            "Attribution-NonCommercial-NoDerivatives 4.0 International License."
        ),
        commercial_ok="no",
        access="refused",
        note=(
            "NoDerivatives is the blocker, not NonCommercial. A training tensor "
            "is a derivative; so is a re-rendered SDR pair. ND forbids it "
            "outright regardless of commercial intent, which makes this "
            "unusable even for internal research. Shame -- it ships matched "
            "BT.2020 SDR and BT.2100 PQ versions of every sequence."
        ),
    ),

    Source(
        sid="red_samples",
        name="RED sample R3D files",
        classes="specular exteriors (L.A. Sunset)",
        scenes="several",
        licence="RED restricted",
        licence_url="https://support.reddigitalcinema.com/hc/en-us/articles/360021857073-Sample-R3D-Files",
        licence_quote=(
            "All footage is provided 'as is'. Use of the footage is for internal "
            "testing purposes only, and may not be used for any other purposes, "
            "whether commercial or non-commercial. RED retains all copyrights "
            "in the footage."
        ),
        commercial_ok="no",
        access="refused",
        note="'Internal testing purposes only' excludes training on either tier.",
    ),

    Source(
        sid="academic_cn",
        name="Real-HDRV / HDRVD2K / HDRTV1K",
        classes="day+night interior and exterior; large volume",
        scenes="500 / 500 / ~20-30",
        licence="academic only",
        licence_url="https://github.com/yungsyu99/Real-HDRV",
        licence_quote=(
            "Our Real-HDRV dataset is available for the academic purpose only. "
            "-- HDRVD2K: 'Our dataset is available for the academic purpose "
            "only.' -- HDRTV1K: 'Please download this dataset only for academic "
            "use.'"
        ),
        commercial_ok="no",
        access="refused",
        note=(
            "Two problems, and the second is worse. Academic-only, and "
            "Baidu-Netdisk-hosted so painful to fetch from Cairo. But HDRVD2K's "
            "500 sources were collected from third-party HDR10 platforms, and "
            "HDRTV1K's from YouTube -- the authors cannot grant rights they do "
            "not hold, so their permission is not worth much even if you took "
            "it. Provenance you cannot defend is worse than data you do not "
            "have."
        ),
    ),

    Source(
        sid="stock_libraries",
        name="Commercial stock (Filmsupply, Artgrid, Shutterstock, Adobe, Getty...)",
        classes="would have supplied everything",
        scenes="thousands",
        licence="explicit AI-training prohibition",
        licence_url="https://www.filmsupply.com/terms-and-conditions",
        licence_quote=(
            "Filmsupply: 'The Service is not intended for and may not be used by "
            "any person, firm, or entity engaged in the development, training, or "
            "operation of generative artificial intelligence systems, models, or "
            "technologies.' -- Shutterstock: 'Use any Visual Content (in whole "
            "or in part) as training data for any artificial intelligence, "
            "machine learning, or generative AI system, tool, process, or "
            "dataset.' (under 'YOU MAY NOT') -- Adobe: 'to directly or "
            "indirectly create, train, test, or otherwise improve any machine "
            "learning algorithms or artificial intelligence systems, including "
            "any architectures, models, or weights.'"
        ),
        commercial_ok="no",
        access="refused",
        note=(
            "The market is closed on terms, not price, and the reason is "
            "commercial: they unbundled training into a separate SKU. "
            "Shutterstock and Storyblocks will sell an ML dataset licence -- "
            "Shutterstock's says datasets 'may only use them to train machine "
            "learning and computer vision models', the inverse of a normal stock "
            "licence. But their ML-licensable libraries are Rec.709 8-bit "
            "delivery transcodes; Shutterstock's own contributor spec says "
            "'REC 709 is preferred'. So the one route that will legally sell us "
            "training rights sells footage that does not contain the signal we "
            "are trying to learn. Clean and useless. Format quality and licence "
            "availability are inversely correlated across the entire market."
        ),
    ),
]

BY_ID = {s.sid: s for s in SOURCES}


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------

def _open(url: str, headers: dict | None = None, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def get_text(url: str) -> str:
    with _open(url) as r:
        return r.read().decode("utf-8", "replace")


def head(url: str) -> tuple[int, int | None, str]:
    """(status, content_length, detail). Falls back to a ranged GET, because a
    few CDNs answer HEAD with 403 and a GET with 206."""
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            cl = r.headers.get("Content-Length")
            return r.status, int(cl) if cl else None, ""
    except urllib.error.HTTPError as e:
        if e.code not in (403, 405, 501):
            return e.code, None, e.reason or ""
    except Exception as e:                                  # noqa: BLE001
        return 0, None, f"{type(e).__name__}: {e}"
    try:
        with _open(url, {"Range": "bytes=0-0"}) as r:
            rng = r.headers.get("Content-Range", "")
            total = int(rng.rsplit("/", 1)[1]) if "/" in rng else None
            return r.status, total, "HEAD refused, ranged GET accepted"
    except urllib.error.HTTPError as e:
        return e.code, None, e.reason or ""
    except Exception as e:                                  # noqa: BLE001
        return 0, None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def parse_s3_page(xml_text: str, include: str) -> tuple[list[dict], str | None]:
    """One ListObjectsV2 page -> ([{key,size,url-less}], continuation token)."""
    root = ET.fromstring(xml_text)
    pat = re.compile(include) if include else None
    out = []
    for c in root.findall(f"{S3_NS}Contents"):
        key = (c.findtext(f"{S3_NS}Key") or "")
        size = int(c.findtext(f"{S3_NS}Size") or 0)
        if not key or key.endswith("/") or size == 0:
            continue
        if pat and not pat.search(key):
            continue
        out.append({"key": key, "size": size})
    truncated = (root.findtext(f"{S3_NS}IsTruncated") or "false").strip().lower() == "true"
    token = root.findtext(f"{S3_NS}NextContinuationToken") if truncated else None
    return out, token


def s3_list(base: str, prefix: str, include: str) -> list[dict]:
    """List one prefix of an S3 or S3-compatible bucket.

    base_url is stated per source rather than derived from a bucket name,
    because the two rules differ and guessing gets it wrong in both directions.
    AWS needs PATH style here -- download.opencontent.netflix.com contains dots,
    so the virtual-host form breaks the wildcard certificate and fails TLS
    before it fails S3. Laval's hdrdb-public.s3.valeria.science is not AWS at
    all and is already a complete hostname. One field, no inference.
    """
    files, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        page, token = parse_s3_page(get_text(base + "?" + urllib.parse.urlencode(q)), include)
        for f in page:
            f["url"] = base + urllib.parse.quote(f["key"])
            files.append(f)
        if not token:
            return files


def polyhaven_list(resolution: str) -> list[dict]:
    ids = sorted(json.loads(get_text("https://api.polyhaven.com/assets?type=hdris")))
    out = []
    for i, aid in enumerate(ids, 1):
        try:
            f = json.loads(get_text(f"https://api.polyhaven.com/files/{aid}"))
            e = f["hdri"][resolution]["exr"]
        except Exception:                                   # noqa: BLE001
            continue
        out.append({"key": f"{aid}_{resolution}.exr", "url": e["url"],
                    "size": e.get("size", 0), "md5": e.get("md5", "")})
        if i % 100 == 0:
            print(f"    polyhaven: {i}/{len(ids)} queried", flush=True)
    return out


def resolve(src: Source) -> list[dict]:
    if src.kind == "s3":
        files = []
        for p in src.prefixes:
            files += s3_list(src.base_url, p, src.include)
        return files
    if src.kind == "static":
        return [{"key": u.rsplit("/", 1)[1], "url": u, "size": 0} for u in src.urls]
    if src.kind == "polyhaven":
        return polyhaven_list(src.resolution)
    return []


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------

def human(n: int | None) -> str:
    if not n:
        return "?"
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024.0
    return "?"


def download(url: str, dest: Path, expect: int = 0) -> dict:
    """Resumable. Hashes the whole file on completion, including bytes a
    previous run wrote -- a resumed file that is never fully hashed is a file
    you cannot pin in a manifest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    t0 = time.time()
    try:
        with _open(url, headers) as r, open(part, "ab" if have else "wb") as fh:
            if have and r.status != 206:
                fh.close()
                part.unlink(missing_ok=True)
                return download(url, dest, expect)      # server ignored Range
            total = have + int(r.headers.get("Content-Length") or 0)
            done = have
            last = 0.0
            while True:
                buf = r.read(CHUNK)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if total and time.time() - last > 5:
                    last = time.time()
                    rate = (done - have) / max(time.time() - t0, 1e-6) / 1048576
                    print(f"      {done*100.0/total:5.1f}%  {human(done)}/{human(total)}"
                          f"  {rate:.1f} MB/s", flush=True)
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code} {e.reason}"}
    except Exception as e:                                  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    h = hashlib.sha256()
    with open(part, "rb") as fh:
        for blk in iter(lambda: fh.read(CHUNK), b""):
            h.update(blk)
    size = part.stat().st_size
    if expect and size != expect:
        return {"ok": False, "error": f"size {size} != expected {expect}",
                "bytes": size, "sha256": h.hexdigest()}
    part.replace(dest)
    return {"ok": True, "bytes": size, "sha256": h.hexdigest(),
            "seconds": round(time.time() - t0, 1)}


def write_licence(src: Source, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "licence.txt").write_text(
        f"{src.name}\n{'=' * len(src.name)}\n\n"
        f"licence        : {src.licence}\n"
        f"licence_url    : {src.licence_url}\n"
        f"commercial_ok  : {src.commercial_ok}\n"
        f"read_at        : {datetime.now(timezone.utc).isoformat()}\n"
        f"independent    : {src.scenes}\n"
        f"corpus_classes : {src.classes}\n\n"
        f"VERBATIM, from the source page:\n\n{src.licence_quote}\n\n"
        f"NOTE\n\n{src.note}\n",
        encoding="utf-8")


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def selected(args) -> list[Source]:
    if args.only:
        missing = [s for s in args.only if s not in BY_ID]
        if missing:
            sys.exit(f"unknown --only: {', '.join(missing)}\n"
                     f"known: {', '.join(BY_ID)}")
        return [BY_ID[s] for s in args.only]
    return [s for s in SOURCES if s.access == "direct" and s.default
            and s.sid not in args.skip]


def mode_plan(args, do_check: bool) -> None:
    srcs = selected(args)
    grand = 0
    for src in srcs:
        if src.access == "refused":
            print(f"\n!! {src.sid} is marked refused -- its licence text forbids this.")
            print(f"   {src.licence_quote[:200]}")
            print("   Not resolving. Edit access= in the source table if that is your call.")
            continue
        print(f"\n=== {src.sid}  ({src.name})")
        print(f"    licence {src.licence} | commercial_ok {src.commercial_ok} "
              f"| scenes {src.scenes}")
        try:
            files = resolve(src)
        except Exception as e:                              # noqa: BLE001
            print(f"    !! could not resolve: {type(e).__name__}: {e}")
            continue
        if not files:
            print("    !! resolved to zero files -- check the include pattern")
            continue
        known = sum(f["size"] for f in files)
        grand += known
        print(f"    {len(files)} files, {human(known)}"
              f"{' (sizes unknown for static URLs)' if not known else ''}")
        for f in files[:8]:
            print(f"      {human(f['size']):>10}  {f['key']}")
        if len(files) > 8:
            print(f"      ... and {len(files) - 8} more")
        if do_check:
            print("    checking:")
            for f in files:
                st, cl, detail = head(f["url"])
                flag = "ok " if st in (200, 206) else "FAIL"
                mism = ""
                if f["size"] and cl and cl != f["size"]:
                    mism = f"  SIZE MISMATCH listed={f['size']} server={cl}"
                    flag = "WARN"
                print(f"      [{flag}] {st:>3} {human(cl):>10}  {f['key']}"
                      f"{('  ' + detail) if detail else ''}{mism}")
    print(f"\nTOTAL to fetch: {human(grand)}")

    todo = [s for s in SOURCES if s.access in ("gated", "paid")]
    if todo:
        print("\n" + "=" * 74)
        print("A PERSON HAS TO DO THESE. I cannot register, accept a EULA, or buy.")
        print("=" * 74)
        for s in todo:
            print(f"\n  {s.sid}  [{s.access}]  {s.scenes} scenes")
            print(f"    {s.name}")
            print(f"    -> {s.action}")
    refused = [s for s in SOURCES if s.access == "refused"]
    if refused:
        print("\n" + "=" * 74)
        print("NOT FETCHED -- the licence text forbids it, in words")
        print("=" * 74)
        for s in refused:
            print(f"  {s.sid:20} {s.licence:32} {s.name}")


def mode_fetch(args) -> None:
    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest_root).free
    print(f"destination {dest_root}  free {human(free)}")

    srcs = [s for s in selected(args) if s.access != "refused"]
    for src in srcs:
        out = dest_root / src.sid
        print(f"\n=== {src.sid}")
        try:
            files = resolve(src)
        except Exception as e:                              # noqa: BLE001
            print(f"    !! resolve failed: {type(e).__name__}: {e}")
            continue
        need = sum(f["size"] for f in files)
        if need and need > free * 0.95:
            print(f"    !! needs {human(need)}, only {human(free)} free. Skipping.")
            continue
        write_licence(src, out)
        man_path = out / "_manifest.json"
        man = json.loads(man_path.read_text()) if man_path.exists() else {}
        for i, f in enumerate(files, 1):
            target = out / f["key"].split("/")[-1]
            rec = man.get(f["key"])
            if rec and rec.get("ok") and target.exists() \
                    and target.stat().st_size == rec.get("bytes"):
                print(f"    [{i}/{len(files)}] have {f['key']}")
                continue
            print(f"    [{i}/{len(files)}] {f['key']}  {human(f['size'])}")
            res = download(f["url"], target, f["size"])
            res.update(url=f["url"], key=f["key"],
                       fetched_at=datetime.now(timezone.utc).isoformat())
            if f.get("md5") and res.get("ok"):
                res["md5_expected"] = f["md5"]
            man[f["key"]] = res
            man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
            if not res.get("ok"):
                print(f"        !! {res.get('error')}")
            free = shutil.disk_usage(dest_root).free
        ok = sum(1 for r in man.values() if r.get("ok"))
        print(f"    {ok}/{len(man)} complete")


def mode_report(args) -> None:
    dest_root = Path(args.dest)
    print(f"{'set':22} {'files':>6} {'ok':>5} {'bytes':>11}  licence / commercial_ok")
    for src in SOURCES:
        man_path = dest_root / src.sid / "_manifest.json"
        if not man_path.exists():
            continue
        man = json.loads(man_path.read_text())
        ok = [r for r in man.values() if r.get("ok")]
        print(f"{src.sid:22} {len(man):6} {len(ok):5} "
              f"{human(sum(r.get('bytes', 0) for r in ok)):>11}  "
              f"{src.licence} / {src.commercial_ok}")
    print("\nNothing here has been measured yet. A scene does not enter the")
    print("training manifest until peak_nits, dr_stops and clipped_fraction at")
    print("0 EV are recorded and it meets its class floor.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=r"G:\datasets\sources")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--fetch", action="store_true")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--only", nargs="*", default=[], metavar="ID")
    ap.add_argument("--skip", nargs="*", default=[], metavar="ID")
    args = ap.parse_args()

    if args.plan or args.check:
        mode_plan(args, do_check=args.check)
    elif args.fetch:
        mode_fetch(args)
    else:
        mode_report(args)


if __name__ == "__main__":
    main()
