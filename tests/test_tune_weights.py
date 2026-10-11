"""Offline tests for the public tuner (eval/tune_weights.py) and the
CINE_REC_WEIGHTS custom-artifact loader."""

import importlib
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import tune_weights as tw  # noqa: E402

from cine_rec_engine import service as svc  # noqa: E402


# ----------------------------------------------------------------- SGD

class TestFit:
    def _rows(self, n=400, dim=None, seed=13):
        dim = dim or len(tw.PUBLIC_FEATURES)
        rng = random.Random(seed)
        w_true = [rng.uniform(-2, 2) for _ in range(dim)]
        rows = []
        for _ in range(n):
            g = [rng.uniform(0, 1) for _ in range(dim)]
            b = [rng.uniform(0, 1) for _ in range(dim)]
            # label BY the true preference — the fit must recover its direction
            if sum(w * x for w, x in zip(w_true, g)) < \
                    sum(w * x for w, x in zip(w_true, b)):
                g, b = b, g
            rows.append((g, b))
        return rows, w_true

    def test_recovers_direction_of_true_weights(self):
        rows, w_true = self._rows()
        w = tw.fit(rows, alpha=0.0, epochs=400, init=[0.0] * len(w_true))
        # pairwise logistic identifies direction, not scale — compare
        # normalized vectors
        def unit(v):
            norm = sum(x * x for x in v) ** 0.5
            return [x / norm for x in v]
        assert sum(a * b for a, b in zip(unit(w), unit(w_true))) > 0.97

    def test_strong_l2_stays_near_init(self):
        rows, _ = self._rows(n=50)
        init = [1.0] * len(tw.PUBLIC_FEATURES)
        w = tw.fit(rows, alpha=1.0, epochs=200, init=list(init))
        assert max(abs(a - b) for a, b in zip(w, init)) < 0.2

    def test_accuracy_on_separable_data(self):
        # one feature fully separates good from bad
        rows = [([1.0] + [0.0] * 21, [0.0] * 22) for _ in range(60)]
        w = tw.fit(rows, alpha=0.0, epochs=100, init=[0.0] * 22)
        assert tw.pairwise_accuracy(w, rows) == 1.0

    def test_accuracy_empty_rows(self):
        assert tw.pairwise_accuracy([1.0] * 22, []) == 0.0


# ------------------------------------------------------------- judgments

def _write_judgments(tmp_path, lines):
    p = tmp_path / "j.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return str(p)


class TestJudgments:
    def _rec(self):
        return {"seed": [155, "movie"],
                "good": [[77, "movie"], [414906, "movie"]],
                "bad": [[634649, "movie"]]}

    def test_parse_and_expand(self, tmp_path):
        path = _write_judgments(tmp_path, [self._rec()])
        judgments = tw.load_judgments(path)
        trips = tw.expand_triplets(judgments)
        assert len(trips) == 2  # 2 good x 1 bad
        assert trips[0] == ((155, "movie"), (77, "movie"), (634649, "movie"))

    def test_missing_field_is_loud(self, tmp_path):
        rec = self._rec()
        del rec["bad"]
        with pytest.raises(SystemExit, match="missing 'bad'"):
            tw.load_judgments(_write_judgments(tmp_path, [rec]))

    def test_bad_media_type_is_loud(self, tmp_path):
        rec = self._rec()
        rec["seed"] = [155, "film"]
        with pytest.raises(SystemExit, match="seed must be"):
            tw.load_judgments(_write_judgments(tmp_path, [rec]))

    def test_empty_file_is_loud(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("\n\n")
        with pytest.raises(SystemExit, match="no judgment lines"):
            tw.load_judgments(str(p))


# ---------------------------------------------------------------- rows

def _ent(id_, mt="movie", year=2000):
    return {
        "id": id_, "media_type": mt, "title": f"t{id_}", "title_en": f"t{id_}",
        "genres": ["Action"], "keywords": ["heist"], "cast_ids": [1, 2],
        "cast_list": ["A", "B"], "director_ids": [5], "writer_ids": [],
        "composer_ids": [], "dp_ids": [], "companies": ["X"],
        "networks": [], "overview": "a bank heist thriller", "overview_en": None,
        "original_language": "en", "release_year": year, "vote_average": 7.0,
        "vote_count": 5000, "collection_id": None, "adult": False,
        "via": "genre", "rating": 7.0, "poster_path": None,
    }


class TestBuildRows:
    def test_rows_are_22_public_features(self):
        mat = {"info": {(1, "movie"): _ent(1), (2, "movie"): _ent(2, year=2010),
                        (3, "movie"): _ent(3, year=1990)},
               "enriched": {}, "sim": {}}
        trips = [((1, "movie"), (2, "movie"), (3, "movie"))]
        rows, dropped = tw.build_rows(trips, mat)
        assert dropped == 0 and len(rows) == 1
        assert len(rows[0][0]) == 22
        assert len(tw.PUBLIC_FEATURES) == 22
        # every name is one the bundled artifact schema knows
        bundled = set(json.loads(
            (Path(svc.__file__).parent / "weights.json").read_text()
        )["feature_names"])
        assert set(tw.PUBLIC_FEATURES) == bundled

    def test_missing_candidate_drops_triplet(self):
        mat = {"info": {(1, "movie"): _ent(1), (2, "movie"): _ent(2)},
               "enriched": {}, "sim": {}}
        trips = [((1, "movie"), (2, "movie"), (99, "movie"))]
        rows, dropped = tw.build_rows(trips, mat)
        assert rows == [] and dropped == 1


class TestSplit:
    def test_holdout_excludes_whole_seeds(self):
        trips = [((s, "movie"), (10, "movie"), (20, "movie")) for s in range(10)]
        rows = [(i, i + 1) for i in range(10)]
        train, test, n_hold = tw.split_by_seed(trips, rows, 0.2, seed=7)
        assert n_hold == 2 and len(test) == 2 and len(train) == 8
        assert len(train) + len(test) == len(trips)


# ------------------------------------------------------------ blend tuning

def _rec(feats21, sims, fallback=0.0, key=None):
    return {"feats21": feats21, "sims": sims, "fallback": fallback,
            "key": key}


class TestDecompose:
    def test_roundtrip_is_exact_algebra(self):
        # w_cos * (sum(beta*s)/sum(beta)) == sum(gamma*s) when all gamma>0
        gamma = [0.6, 0.3, 0.1]
        cols = ["embedding_a", "embedding_b", "embedding_c"]
        beta, w_cos = tw.decompose(gamma, cols)
        s = [0.7, 0.4, 0.9]
        lhs = w_cos * sum(b * x for b, x in zip(beta.values(), s)) / sum(beta.values())
        rhs = sum(g * x for g, x in zip(gamma, s))
        assert abs(lhs - rhs) < 1e-12

    def test_negative_gamma_clipped_and_dropped(self):
        beta, w_cos = tw.decompose([0.5, -0.2, 0.0], ["a", "b", "c"])
        assert list(beta) == ["a"]
        assert beta["a"] == 0.5


class TestDeployableCosine:
    def test_renormalizes_over_seed_active_columns(self):
        beta = {"a": 2.0, "b": 1.0}
        rec = _rec([], {"a": 0.8, "b": 0.5})
        seed_has = {"a": {("s", "movie")}, "b": set()}  # seed holds only a
        got = tw.deployable_cosine(rec, ("s", "movie"), beta,
                                   ["a", "b"], seed_has)
        assert got == 0.8  # renormalized away from the missing column

    def test_candidate_missing_seed_column_falls_back(self):
        beta = {"a": 2.0, "b": 1.0}
        rec = _rec([], {"a": 0.8, "b": None}, fallback=0.31)
        seed_has = {"a": {("s", "movie")}, "b": {("s", "movie")}}
        got = tw.deployable_cosine(rec, ("s", "movie"), beta,
                                   ["a", "b"], seed_has)
        assert got == 0.31  # engine skips the pair -> text overlap

    def test_empty_beta_falls_back(self):
        rec = _rec([], {"a": 0.8}, fallback=0.2)
        got = tw.deployable_cosine(rec, ("s", "movie"), {}, ["a"], {})
        assert got == 0.2


class TestBlendEndToEnd:
    def test_informative_column_gets_the_weight(self):
        # two columns: col 'good' separates the pairs, col 'noise' is random
        rng = random.Random(3)
        records = []
        for i in range(120):
            g_good, b_good = 0.9, 0.1
            g_noise, b_noise = rng.random(), rng.random()
            records.append((
                ("s%d" % (i % 8), "movie"),
                _rec([0.0] * 21, {"embedding_good": g_good,
                                  "embedding_noise": g_noise}),
                _rec([0.0] * 21, {"embedding_good": b_good,
                                  "embedding_noise": b_noise}),
            ))
        cols = ["embedding_good", "embedding_noise"]
        h21 = tw._heuristic21()
        init = h21 + [1.0, 1.0]
        fitted = tw.fit(tw.expand_rows(records, cols), alpha=0.0,
                        epochs=400, init=init, dim=len(init))
        beta, w_cos = tw.decompose(fitted[len(h21):], cols)
        assert beta.get("embedding_good", 0) > beta.get("embedding_noise", 0)
        acc = tw.blend_model_accuracy(
            records, beta, cols, fitted[:len(h21)], w_cos,
            {"embedding_good": {r[0] for r in records},
             "embedding_noise": {r[0] for r in records}})
        assert acc == 1.0


class TestBlendRecords:
    def test_build_from_canned_mat(self):
        mat = {
            "info": {(1, "movie"): _ent(1), (2, "movie"): _ent(2, year=2010),
                     (3, "movie"): _ent(3, year=1990)},
            "enriched": {}, "sim": {},
            "sim_cols": {"embedding_a": {((1, "movie"), (2, "movie")): 0.8,
                                         ((1, "movie"), (3, "movie")): 0.2}},
            "seed_has_col": {"embedding_a": {(1, "movie")}},
        }
        trips = [((1, "movie"), (2, "movie"), (3, "movie"))]
        records, dropped, cols = tw.build_blend_records(
            trips, mat, ["embedding_a"])
        assert dropped == 0 and cols == ["embedding_a"]
        s, g, b = records[0]
        assert len(g["feats21"]) == 21
        assert g["sims"]["embedding_a"] == 0.8
        assert g["fallback"] >= 0.0  # text-overlap value fills the slot

    def test_missing_candidate_drops(self):
        mat = {"info": {(1, "movie"): _ent(1), (2, "movie"): _ent(2)},
               "enriched": {}, "sim": {}, "sim_cols": {},
               "seed_has_col": {}}
        trips = [((1, "movie"), (2, "movie"), (99, "movie"))]
        records, dropped, cols = tw.build_blend_records(trips, mat, [])
        assert records == [] and dropped == 1


class TestBlendColumnsValidation:
    def test_rejects_non_identifier(self):
        import asyncio
        with pytest.raises(SystemExit, match="invalid blend column"):
            asyncio.run(tw.discover_blend_columns(None, override='bad"col'))


# ------------------------------------------------- CINE_REC_WEIGHTS loader

@pytest.fixture(autouse=True)
def _restore_scorer(monkeypatch):
    import cine_rec_engine.config as cfg

    blend_before = dict(cfg.COSINE_BLEND)
    yield
    monkeypatch.delenv("CINE_REC_SCORER", raising=False)
    monkeypatch.delenv("REC_SCORER", raising=False)
    monkeypatch.delenv("CINE_REC_WEIGHTS", raising=False)
    cfg.COSINE_BLEND = blend_before
    importlib.reload(svc)


class TestCustomWeightsLoader:
    def test_custom_artifact_loads_and_blends(self, monkeypatch, tmp_path):
        art = tmp_path / "my.json"
        art.write_text(json.dumps({
            "model": "pairwise-logistic-sgd", "alpha": 1e-4, "triplets": 300,
            "feature_names": ["cosine_sim"],
            "weights": {"cosine_sim": 9.5, "bogus_feature": 1.0},
        }))
        monkeypatch.setenv("CINE_REC_SCORER", "learned")
        monkeypatch.setenv("CINE_REC_WEIGHTS", str(art))
        mod = importlib.reload(svc)
        assert mod.LEARNED_WEIGHTS["cosine_sim"] == 9.5          # custom wins
        assert mod.LEARNED_WEIGHTS["keyword_sim"] == \
            mod.HEURISTIC_WEIGHTS["keyword_sim"]                 # heuristic fills
        assert "bogus_feature" not in mod.LEARNED_WEIGHTS        # unknown dropped
        assert str(art) in mod.SCORER_SOURCE

    def test_broken_custom_path_falls_back_loudly(self, monkeypatch, caplog):
        monkeypatch.setenv("CINE_REC_SCORER", "learned")
        monkeypatch.setenv("CINE_REC_WEIGHTS", "/nonexistent/weights.json")
        mod = importlib.reload(svc)
        assert mod.LEARNED_WEIGHTS is None
        assert mod.ACTIVE_WEIGHTS is mod.HEURISTIC_WEIGHTS
        assert mod.SCORER_SOURCE == "heuristic"

    def test_learned_without_custom_path_is_bundled(self, monkeypatch):
        monkeypatch.setenv("CINE_REC_SCORER", "learned")
        monkeypatch.delenv("CINE_REC_WEIGHTS", raising=False)
        mod = importlib.reload(svc)
        assert "bundled" in mod.SCORER_SOURCE

    def test_artifact_cosine_blend_is_applied(self, monkeypatch, tmp_path):
        import cine_rec_engine.config as cfg

        art = tmp_path / "my.json"
        art.write_text(json.dumps({
            "weights": {"cosine_sim": 9.5},
            "feature_names": ["cosine_sim"],
            "cosine_blend": {"embedding_e5e": 0.8, "embedding_v4": 0.2,
                             "embedding_bad": 0.0},
        }))
        monkeypatch.setenv("CINE_REC_SCORER", "learned")
        monkeypatch.setenv("CINE_REC_WEIGHTS", str(art))
        importlib.reload(svc)
        # positive weights flow to the live module attribute; zero dropped
        assert cfg.COSINE_BLEND == {"embedding_e5e": 0.8, "embedding_v4": 0.2}

    def test_artifact_without_blend_leaves_config_alone(self, monkeypatch, tmp_path):
        import cine_rec_engine.config as cfg

        art = tmp_path / "my.json"
        art.write_text(json.dumps(
            {"weights": {"cosine_sim": 9.5}, "feature_names": ["cosine_sim"]}))
        monkeypatch.setenv("CINE_REC_SCORER", "learned")
        monkeypatch.setenv("CINE_REC_WEIGHTS", str(art))
        before = dict(cfg.COSINE_BLEND)
        importlib.reload(svc)
        assert cfg.COSINE_BLEND == before

    def test_custom_ignored_without_learned_gate(self, monkeypatch, tmp_path):
        art = tmp_path / "my.json"
        art.write_text(json.dumps(
            {"weights": {"cosine_sim": 9.5}, "feature_names": ["cosine_sim"]}))
        monkeypatch.delenv("CINE_REC_SCORER", raising=False)
        monkeypatch.setenv("CINE_REC_WEIGHTS", str(art))
        mod = importlib.reload(svc)
        assert mod.LEARNED_WEIGHTS is None
        assert mod.SCORER_SOURCE == "heuristic"
