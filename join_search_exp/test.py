from ckan import CanadaCKAN

if __name__ == '__main__':
    import os
    import shutil

    ckan_client = CanadaCKAN()         
    
    if os.path.exists('tmp'):
        shutil.rmtree('tmp')
    os.makedirs('tmp', exist_ok=True)

    cnt = ckan_client.download_tables_from_package_search('tmp', 'csv', 3, q='suppliers+payments', rows=20, verbose=True)

