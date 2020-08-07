from lib.jupytools import iprint


def dump(ds):
    sql = """
    select
        s.rowid,
        s.name,
        s.file_id,
        a.hive,
        l.nobee,
        iif(l.nobee < 0.5, 0, 1) as b_nobee, -- using sqlite builtin function
        l.queen
    from samples s, attributes a, labels l
    where a.sample_id = s.rowid
    and l.sample_id = s.rowid
    and a.hive = 'Hive1'
    """
    iprint("It works")
    return ds.getDataFrame(sql)
