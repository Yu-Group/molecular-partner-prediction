import data


def test_pipeline():
    '''test that the data pipeline succesfully completes
    '''
    print('testing pipeline...')
    df = data.get_data()
    assert (df.lifetime.max() < 300)


if __name__ == '__main__':
    test_pipeline()
    print('all tests passed!')
