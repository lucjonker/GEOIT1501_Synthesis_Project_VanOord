import geopandas
from sqlalchemy import create_engine

db_connection_url = "postgresql://postgres.miiedebavuhxxbzpndeq:SYbFFBRcyttS3XQy@aws-1-eu-west-3.pooler.supabase.com:5432/postgres"
con = create_engine(db_connection_url)
sql = "SELECT scid, geom, name FROM side_channels"
df = geopandas.read_postgis(sql, con, index_col="scid")
df.sort_index(inplace=True)
print(df)