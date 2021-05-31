CREATE TABLE STRATEGY(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy CHAR(4) NOT NULL,
    cnt INTEGER DEFAULT 0,
    status CHAR(7) DEFAULT "READY",
    v_max_score FLOAT DEFAULT 0.0,
    v_avg_score FLOAT DEFAULT 0.0,
    lb_max_score FLOAT DEFAULT 0.0,
    lb_avg_score FLOAT DEFAULT 0.0
);

CREATE TABLE MODEL(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy CHAR(4) NOT NULL,
    ms_name VARCHAR(50) NOT NULL,
    s_cnt INTEGER NOT NULL,
    v_score FLOAT DEFAULT 0.0,
    lb_score FLOAT DEFAULT 0.0,
    dataset CHAR(10) DEFAULT "dataset_v1",
    dataset_idx CHAR(1) DEFAULT "0"
);

INSERT INTO STRATEGY(strategy) VALUES('st03');
INSERT INTO STRATEGY(strategy) VALUES('st04');
INSERT INTO STRATEGY(strategy) VALUES('st05');
