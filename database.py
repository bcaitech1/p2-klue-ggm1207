""" db와의 연결을 계속 유지할까? 말까? 말자."""

import sqlite3


def db_connect(dbname="./database/nopain.db"):
    conn = sqlite3.connect(dbname)
    return conn


def execute_query(query):
    conn = db_connect()

    cur = conn.cursor()
    cur.execute(query)

    conn.commit()
    conn.close()


def get_scores_of_strategy(strategy):
    """ 전략이 들어갔다면 무조건 Default 값이라도 뽑아낸다. """
    conn = db_connect()
    query = f"SELECT cnt, v_max_score, v_avg_score FROM STRATEGY WHERE strategy='{strategy}'"

    cur = conn.cursor()
    cur.execute(query)

    rows = cur.fetchone()  # fetchone, fetchmany
    conn.close()

    return rows


def get_scores():
    """ 전략이 들어갔다면 무조건 Default 값이라도 뽑아낸다. """
    conn = db_connect()
    query = "SELECT cnt, v_max_score, v_avg_score FROM STRATEGY"

    cur = conn.cursor()
    cur.execute(query)

    rows = cur.fetchone()  # fetchone, fetchmany
    conn.close()

    return rows


def insert_model_scores(args, s_cnt, acc):
    query = f"INSERT INTO MODEL(strategy, ms_name, s_cnt, v_score, dataset, dataset_idx) \
VALUES('{args.strategy}', '{args.ms_name}', {int(s_cnt)}, {acc}, '{args.data_kind}', '{args.dataset_idx}')"
    execute_query(query)


def update_strategy_statistics(args, valid_acc):
    """ 평균 계산하고, MAX_SCORE 계산 """
    flag = False
    prev_cnt, prev_max_score, prev_avg_score = get_scores_of_strategy(args.strategy)

    if valid_acc > prev_max_score:
        prev_max_score = valid_acc

    new_avg_score = prev_avg_score * prev_cnt + valid_acc
    new_avg_score /= prev_cnt + 1

    query = f"UPDATE STRATEGY SET v_max_score={prev_max_score}, v_avg_score={new_avg_score} WHERE strategy='{args.strategy}'"
    execute_query(query)

    query = f"UPDATE STRATEGY SET cnt = {prev_cnt+1} WHERE strategy = '{args.strategy}'"
    execute_query(query)

    prev_cnt, all_max_score, prev_avg_score = get_scores()

    if valid_acc > all_max_score:
        flag = True

    return flag


def sample_strategy():
    """ PENDING되지 않은 전략 중, """
    conn = db_connect()
    query = "SELECT strategy, status, cnt, v_avg_score FROM STRATEGY WHERE status is not 'PENDING'"  # 오름차순

    cur = conn.cursor()
    cur.execute(query)

    row_tuple = cur.fetchall()  # fetchone, fetchmany

    conn.close()

    row_tuple = sorted(row_tuple, key=lambda x: (x[2], -x[3]))
    strategy, status, cnt, v_avg_score = row_tuple[0]

    return strategy, status, cnt, v_avg_score
