from SoccerNet.Downloader import SoccerNetDownloader as SNdl

mySNdl = SNdl(LocalDirectory="./data/SoccerNet")
mySNdl.downloadDataTask(task="jersey-2023", split=["train","test","challenge"])