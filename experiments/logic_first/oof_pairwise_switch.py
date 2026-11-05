"""Out-of-fold pairwise candidate switcher; research only.

For each outer grouped fold, train the switcher exclusively on the other folds.
Within its allowed training records, generate training predictions from two
inner SOURCE-GROUP held-out base models (not in-sample predictions).
Only learn cases where one top-two candidate decodes to correct Unicode.
No test label, filename, source ID or competitor prediction at inference.
"""
from __future__ import annotations
import hashlib,math,unicodedata
from collections import Counter
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from codeccat.vnext_l3_language import Rerank,_decode
from architecture_attribution import fit
from vnext_residual_study import residual_subset

THRESHOLD=0.80
MAX_MARGIN=0.50

def quality(text):
    """Script and Unicode quality, independent of expected encoding label."""
    if not text:return np.zeros(8,dtype=float)
    n=len(text)
    c=Counter(unicodedata.category(ch)[0] for ch in text)
    alpha=sum(ch.isalpha() for ch in text)
    whitespace=sum(ch.isspace() for ch in text)
    bad=sum((unicodedata.category(ch) in ("Cn","Co","Cs") or
             (0x80<=ord(ch)<=0x9f)) for ch in text)
    latin=sum("LATIN" in unicodedata.name(ch,"") for ch in text)
    cyrillic=sum("CYRILLIC" in unicodedata.name(ch,"") for ch in text)
    other_script=alpha-latin-cyrillic
    return np.array([alpha/n,whitespace/n,bad/n,c["P"]/n,c["N"]/n,
                     latin/n,cyrillic/n,max(0,other_script)/n],dtype=float)

def features(ranked,raw,language):
    if len(ranked.candidates)<2 or len(ranked.scores)<2:return None
    a,b=ranked.candidates[:2]
    text_a=_decode(raw,a,final=True)
    text_b=_decode(raw,b,final=True)
    if text_a is None or text_b is None or text_a==text_b:return None
    lang_a=language.score(text_a)
    lang_b=language.score(text_b)
    qa=quality(text_a)
    qb=quality(text_b)
    # B relative to A, plus input length and L2 score difference. Base
    # candidates themselves do NOT enter features (no codec memorization).
    return np.concatenate([np.array([
        float(ranked.scores[0]-ranked.scores[1]),
        np.log1p(len(raw))/np.log(4097),
        0.0 if lang_a is None or lang_b is None else
            max(-5.0,min(5.0,lang_b-lang_a)),
        float(lang_a is not None and lang_b is not None)
    ]),(qb-qa)])

def group_bucket(group):
    return int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%2

@dataclass
class LearnedSwitch:
    scaler:object|None
    classifier:object|None
    training_info:dict
    def rerank(self,ranked,raw,*,final=True,max_l2_margin=.5,min_language_advantage=.15):
        # LanguageCoherence-compatible signature; the model stores learned
        # feature scaler but caller supplies its matching L3 language model.
        raise RuntimeError("Use wrap(language).rerank() to supply matching language model")
    def wrap(self,language):
        return BoundSwitch(self,language)

class BoundSwitch:
    def __init__(self,learned,language):
        self.learned=learned;self.language=language
    def rerank(self,ranked,raw,*,final=True,max_l2_margin=.5,min_language_advantage=.15):
        original=self.language.rerank(ranked,raw,final=final,
            max_l2_margin=max_l2_margin,min_language_advantage=min_language_advantage)
        # Preserve any positive existing L3 choice and L1 hard singletons.
        if original.changed or self.learned.classifier is None:return original
        if ranked.encoding is None or len(ranked.scores)<2:return original
        margin=ranked.scores[0]-ranked.scores[1]
        if not 0<=margin<=MAX_MARGIN:return original
        vals=features(ranked,raw,self.language)
        if vals is None:return original
        x=self.learned.scaler.transform([vals])
        p=self.learned.classifier.predict_proba(x)[0,1]
        if p<THRESHOLD:return original
        return Rerank(ranked.candidates[1],True,True,"oof-learned-switch",float(p))

def fit_oof(training,LanguageCoherence,strict):
    groups={group_bucket(g) for b,y,g in training}
    X=[];Y=[];counter=Counter()
    if groups!={0,1}:return LearnedSwitch(None,None,{"reason":"inner-groups-imbalanced"})
    for part in (0,1):
        allowed=[(b,y,g) for b,y,g in training if group_bucket(g)!=part]
        held=[(b,y,g) for b,y,g in training if group_bucket(g)==part]
        if len(set(y for b,y,g in allowed))<2:continue
        mdl=fit(residual_subset(allowed)[0],"binary_high")
        lang=LanguageCoherence.fit(allowed)
        for raw,truth,g in held:
            ranked=mdl.rank(raw,final=True)
            f=features(ranked,raw,lang)
            if f is None:continue
            expected=strict(raw,truth)
            yes_a=strict(raw,ranked.candidates[0])==expected
            yes_b=strict(raw,ranked.candidates[1])==expected
            if yes_a==yes_b:continue
            X.append(f);Y.append(int(yes_b and not yes_a))
            counter["eligible"]+=1;counter["switch_needed"]+=int(yes_b and not yes_a)
        print("OOF_INNER_DONE",part,len(held),flush=True)
    if len(set(Y))<2 or len(Y)<20:
        return LearnedSwitch(None,None,{"reason":"insufficient-discordant-OOF","eligible":len(Y)})
    scaler=StandardScaler()
    scaled=scaler.fit_transform(np.asarray(X))
    clf=LogisticRegression(C=0.1,max_iter=1000,random_state=20261105)
    clf.fit(scaled,Y)
    return LearnedSwitch(scaler,clf,{"eligible":len(Y),"need_switch":sum(Y),
        "training_groups":len(set(g for b,y,g in training)),
        "inner_partitions":2,
        "method":"source-group-disjoint inner OOF"})
