-- Data Factory annotation store -- schema version 1.
--
-- One SQLite file per workspace, sitting *beside* the dataset sandboxes and
-- never inside one, so it is never swept, never zipped by `make_zip`, and
-- never served. See store.py for why SQLite and not files or Postgres.
--
-- Two rules this schema exists to enforce, both of which are correctness
-- issues rather than conveniences:
--
--   1. A label's index is immutable, never reused and never reordered. Class
--      indices leak into every YOLO .txt ever exported, so renumbering them
--      silently re-labels work that is already on disk.
--   2. Everything is reachable only through a project, and a project has an
--      owner. Scoping added later is a rewrite; scoping written now is a JOIN.

-- ---------------------------------------------------------------- projects

CREATE TABLE IF NOT EXISTS project (
    id              TEXT PRIMARY KEY,
    owner_id        TEXT NOT NULL DEFAULT 'local',
    name            TEXT NOT NULL,
    -- managed: the database is the truth, sidecars are produced by export.
    -- attached: a folder on disk the user opened; every save writes through.
    storage         TEXT NOT NULL DEFAULT 'managed',
    -- keep: this is someone's work. ttl: an anonymous upload, swept after a day.
    retention       TEXT NOT NULL DEFAULT 'keep',
    -- Only meaningful for attached storage: the folder being annotated.
    root            TEXT NOT NULL DEFAULT '',
    -- Monotonic label-index allocator. Deliberately NOT max(idx)+1: retiring
    -- the last label and adding another must not hand out the retired index.
    next_label_idx  INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS project_owner ON project(owner_id, created_at DESC);

-- ------------------------------------------------------------------ labels

CREATE TABLE IF NOT EXISTS label (
    project_id  TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    name        TEXT NOT NULL,
    colour      TEXT NOT NULL,          -- '#rrggbb', assigned server-side
    -- Retiring hides a label from the palette without freeing its index, and
    -- without touching shapes already drawn with it.
    retired_at  REAL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (project_id, idx)
);

-- Names are unique among *live* labels only, so a retired name can come back
-- later -- as a new index, which is the whole point.
CREATE UNIQUE INDEX IF NOT EXISTS label_live_name
    ON label(project_id, name) WHERE retired_at IS NULL;

-- ------------------------------------------------------------------ images

CREATE TABLE IF NOT EXISTS image (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    -- POSIX-relative to the project root, so a database made on Windows opens
    -- on Linux. Never absolute: an absolute path in a row is a path a client
    -- could eventually influence.
    rel_path    TEXT NOT NULL,
    -- Oriented dimensions -- what the pixels actually are after EXIF is
    -- applied, which is what the browser draws and what boxes are relative to.
    width       INTEGER NOT NULL DEFAULT 0,
    height      INTEGER NOT NULL DEFAULT 0,
    seq         INTEGER NOT NULL DEFAULT 0,
    -- todo -> predicted -> in_progress -> reviewed, plus skipped ("I can't
    -- tell"), which must never quietly become training data.
    state       TEXT NOT NULL DEFAULT 'todo',
    -- Bumped on every save. A save carries the version it was based on, so a
    -- second tab editing the same image gets a 409 instead of clobbering.
    version     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS image_path ON image(project_id, rel_path);
CREATE INDEX IF NOT EXISTS image_seq ON image(project_id, seq);
-- "the next unreviewed image" is the single hottest query in the editor.
CREATE INDEX IF NOT EXISTS image_state ON image(project_id, state, seq);

-- ------------------------------------------------------------------ shapes

CREATE TABLE IF NOT EXISTS shape (
    id          TEXT PRIMARY KEY,
    image_id    TEXT NOT NULL REFERENCES image(id) ON DELETE CASCADE,
    project_id  TEXT NOT NULL,
    label_idx   INTEGER NOT NULL,
    -- Box columns are real columns, not JSON, because size distribution and
    -- geometry-error analysis are queries, not scans.
    xmin        INTEGER NOT NULL,
    ymin        INTEGER NOT NULL,
    xmax        INTEGER NOT NULL,
    ymax        INTEGER NOT NULL,
    polygon     TEXT,                   -- JSON [[x, y], ...], or NULL
    -- human | model. Model shapes render dashed until someone touches them.
    origin      TEXT NOT NULL DEFAULT 'human',
    confidence  REAL,
    pose        TEXT NOT NULL DEFAULT 'Unspecified',
    truncated   INTEGER NOT NULL DEFAULT 0,
    difficult   INTEGER NOT NULL DEFAULT 0,
    occluded    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    -- Set when a person edits a model shape. That is what turns it solid, and
    -- what the agreement metric counts.
    edited_at   REAL,
    FOREIGN KEY (project_id, label_idx) REFERENCES label(project_id, idx)
);

CREATE INDEX IF NOT EXISTS shape_image ON shape(image_id);
CREATE INDEX IF NOT EXISTS shape_label ON shape(project_id, label_idx);

-- -------------------------------------------------------------- model runs

CREATE TABLE IF NOT EXISTS model_run (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT '',
    matched     INTEGER NOT NULL DEFAULT 0,
    unmatched   INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS model_run_project ON model_run(project_id, created_at DESC);

-- ------------------------------------------------------------------- audit

CREATE TABLE IF NOT EXISTS audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL,
    owner_id    TEXT NOT NULL,
    action      TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '',
    at          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS audit_project ON audit(project_id, at DESC);

-- --------------------------------------------------- accounts (Phase 5)
--
-- Created now, unused now. They cost nothing empty, and having the tables
-- present means Phase 5 adds routes rather than a migration across live data.

CREATE TABLE IF NOT EXISTS user (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL,
    password_hash TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS user_email ON user(email);

CREATE TABLE IF NOT EXISTS session (
    token      TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES user(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS member (
    project_id TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    user_id    TEXT NOT NULL REFERENCES user(id) ON DELETE CASCADE,
    role       TEXT NOT NULL DEFAULT 'owner',
    created_at REAL NOT NULL,
    PRIMARY KEY (project_id, user_id)
);
