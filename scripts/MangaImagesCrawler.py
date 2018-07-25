import logging
import os,os.path as osp
import cv2

from icrawler.builtin import (BaiduImageCrawler, BingImageCrawler,
                              GoogleImageCrawler, GreedyImageCrawler,
                              UrlListCrawler)

test_dir = 'C:/Users/Michal/Documents/Hackaton2018/Datasets/Manga faces/'


def test_google():
    img_dir = test_dir
    google_crawler = GoogleImageCrawler(
        downloader_threads=2,
        storage={'root_dir': img_dir},
        log_level=logging.INFO)
    search_filters = dict(
        license='commercial,modify')
    google_crawler.crawl('color manga face', filters=search_filters, max_num=500)


def test_bing():
    img_dir = osp.join(test_dir, 'bing2')
    bing_crawler = BingImageCrawler(
        downloader_threads=2,
        storage={'root_dir': img_dir},
        log_level=logging.INFO)
    search_filters = dict(
        type='photo',
        license='commercial')
    bing_crawler.crawl('manga face color', max_num=1000, filters=search_filters)


def test_baidu():
    img_dir = osp.join(test_dir, 'baidu')
    search_filters = dict(size='large', color='blue')
    baidu_crawler = BaiduImageCrawler(
        downloader_threads=2, storage={'root_dir': img_dir})
    baidu_crawler.crawl('color manga face', max_num=500)


def replaceNameInFolder(folderName, prefix):
    folder_images = [f for f in os.listdir(folderName)]
    for f in folder_images:
        im = cv2.imread(folderName + "/" + f)
        im_resize = cv2.resize(im, (256, 256))
        cv2.imwrite(folderName + "/" + prefix + "_" + f, im_resize)

if __name__ == "__main__":
    #test_google()
    #test_bing()
    #test_baidu()
    folderName = "C:/Users/Michal/Documents/Hackaton2018/Datasets/Manga faces/googleAndBing"
    replaceNameInFolder(folderName, "p")