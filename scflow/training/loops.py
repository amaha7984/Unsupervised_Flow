def infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x
