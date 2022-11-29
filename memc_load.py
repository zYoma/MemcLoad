#!/usr/bin/env python
# -*- coding: utf-8 -*-
import concurrent.futures
import multiprocessing as mp
import optparse
import os
import gzip
import sys
import glob
import logging
import collections
import time
from asyncio import Future
from concurrent.futures import ProcessPoolExecutor
from functools import wraps
from optparse import OptionParser
from typing import Any, Callable, Generator

# brew install protobuf
# protoc  --python_out=. ./appsinstalled.proto
# pip install protobuf
import appsinstalled_pb2
# pip install python-memcached
import memcache

NORMAL_ERR_RATE = 0.01
PROCESS_CHUNK_SIZE = 1000000
MAX_RETRIES = 3
AppsInstalled = collections.namedtuple("AppsInstalled", ["dev_type", "dev_id", "lat", "lon", "apps"])


def dot_rename(path):
    head, fn = os.path.split(path)
    # atomic in most cases
    os.rename(path, os.path.join(head, "." + fn))


def trying(func):
    """ Декоратор для повторных попыток запуска задач в случае ошибок. Делает max_retries попыток.
        Постоянно удваивает паузу между попытками.
    """
    @wraps(func)
    def wrap(self, *args, **kwargs):
        max_retries = MAX_RETRIES
        count = 0
        while True:
            try:
                return func(self, *args, **kwargs)
            except Exception as err:
                count += 1
                if count > max_retries:
                    logging.exception(err)
                    raise err

                backoff = count * 2
                logging.error('Retrying in {} seconds'.format(backoff))
                time.sleep(backoff)
    return wrap


@trying
def save_data(memc_addr: str, key: str, packed: str) -> None:
    memc = memcache.Client([memc_addr])
    memc.set(key, packed)


def insert_appsinstalled(data, options):
    device_memc = {
        "idfa": options.idfa,
        "gaid": options.gaid,
        "adid": options.adid,
        "dvid": options.dvid,
    }

    dry_run = options.dry
    results = []
    errors = 0
    for appsinstalled in data:
        memc_addr = device_memc.get(appsinstalled.dev_type)
        if not memc_addr:
            errors += 1
            logging.error("Unknow device type: %s" % appsinstalled.dev_type)
            continue

        ua = appsinstalled_pb2.UserApps()
        ua.lat = appsinstalled.lat
        ua.lon = appsinstalled.lon
        key = "%s:%s" % (appsinstalled.dev_type, appsinstalled.dev_id)
        ua.apps.extend(appsinstalled.apps)
        packed = ua.SerializeToString()
        try:
            if dry_run:
                logging.debug("%s - %s -> %s" % (memc_addr, key, str(ua).replace("\n", " ")))
            else:
                save_data(memc_addr, key, packed)
        except Exception as err:
            logging.exception("Cannot write to memc %s: %s" % (memc_addr, err))
            results.append(False)
        results.append(True)

    return results, errors


def parse_appsinstalled(chunk, *args):
    results = []
    errors = 0
    for line in chunk:
        line_parts = line.strip().split("\t")
        if len(line_parts) < 5:
            errors += 1
            continue
        dev_type, dev_id, lat, lon, raw_apps = line_parts
        if not dev_type or not dev_id:
            errors += 1
            continue
        try:
            apps = [int(a.strip()) for a in raw_apps.split(",")]
        except ValueError:
            apps = [int(a.strip()) for a in raw_apps.split(",") if a.isidigit()]
            logging.info("Not all user apps are digits: `%s`" % line)
        try:
            lat, lon = float(lat), float(lon)
        except ValueError:
            logging.info("Invalid geo coords: `%s`" % line)
        results.append(AppsInstalled(dev_type, dev_id, lat, lon, apps))

    return results, errors


def read_file_line(file: gzip, chunk_size: int) -> Generator[list[str], None, None]:
    with gzip.open(file) as f:  # type: ignore
        chunk = []
        counter = chunk_size
        for line in f:
            line = line.decode().strip()
            if not line:
                continue
            chunk.append(line)

            counter -= 1
            if counter == 0:
                # когда чанк наполнился, возвращаем его и очищаем
                yield chunk
                chunk = []
                counter = chunk_size
        # возвращаем последний чанк
        yield chunk


def get_slice_size_and_remainder(lines_count: int, worker_count: int) -> tuple[int, int]:
    """
    Для параллельного парсинга данных делим лог на чанки. Функция определяет размер среза и остаток.
    :param lines_count: всего строк в логе
    :param worker_count: число процессов
    :return: размер чанка и остаток
    """
    slice_size = lines_count // worker_count
    remainder = lines_count % worker_count
    assert slice_size * worker_count + remainder == lines_count
    return slice_size, remainder


def data_preparation_for_processing(
        executor: ProcessPoolExecutor,
        log_file_list: list,
        worker_count: int,
        func: Callable,
        option: optparse.Values,
) -> dict[Future[list[tuple[Any]]], str]:
    """ Функция нарезает исходный лог файл на чанки для паралельного исполнения."""
    # определяем размер чанков для равномерного распределения по процессам
    slice_size, remainder = get_slice_size_and_remainder(len(log_file_list), worker_count)

    slice_start = 0
    slice_end = slice_size
    future_to_pars = {}
    for i in range(worker_count):
        if i + 1 == worker_count:
            # Если это последний чанк, добавляем остаток строк
            slice_end += remainder + 1
        log_slice = log_file_list[slice_start:slice_end]
        future_to_pars[executor.submit(func, log_slice, option)] = log_slice

        slice_start = slice_end
        slice_end += slice_size

    return future_to_pars


def run_in_process_pool(
        func: Callable,
        data: list,
        worker_count: int,
        options: optparse.Values
) -> type[list[Any], int]:
    # запускает функцию с переданными аргументами в отдельном процессе
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_pars = data_preparation_for_processing(
            executor,
            data,
            worker_count,
            func,
            options
        )

    results = []
    errors = 0
    for future in concurrent.futures.as_completed(future_to_pars):
        try:
            pars_data, errors = future.result()
        except Exception as err:
            logging.exception(err)
        else:
            if pars_data:
                results += pars_data
            if errors:
                errors += errors

    return results, errors


def main(options):
    worker_count = mp.cpu_count()
    for fn in glob.iglob(options.pattern):
        processed = errors = 0
        logging.info('Processing %s' % fn)
        # Идея следующая: чтобы не читать полностью весь файл в память, будем получать
        # список строк чанками от генератора. Зная размер каждого чанка и число воркеров,
        # делим чанки поровну между всеми воркерами. Конкурентно обрабатываем несколькими процессами.
        for n, chunk in enumerate(read_file_line(fn, PROCESS_CHUNK_SIZE)):
            logging.info(f'CHUNK {n} size: {len(chunk)} lines')
            # парсим строки
            pars_data, err = run_in_process_pool(parse_appsinstalled, chunk, worker_count, options)
            if err:
                errors += err
            # загружаем данные в memcache
            results, err = run_in_process_pool(insert_appsinstalled, pars_data, worker_count, options)
            if err:
                errors += err

            for ok in results:
                if ok:
                    processed += 1
                else:
                    errors += 1

            if not processed:
                dot_rename(fn)
                continue

        err_rate = float(errors) / processed
        if err_rate < NORMAL_ERR_RATE:
            logging.info("Acceptable error rate (%s). Successfull load" % err_rate)
        else:
            logging.error("High error rate (%s > %s). Failed load" % (err_rate, NORMAL_ERR_RATE))
        dot_rename(fn)


def prototest():
    sample = "idfa\t1rfw452y52g2gq4g\t55.55\t42.42\t1423,43,567,3,7,23\ngaid\t7rfw452y52g2gq4g\t55.55\t42.42\t7423,424"
    for line in sample.splitlines():
        dev_type, dev_id, lat, lon, raw_apps = line.strip().split("\t")
        apps = [int(a) for a in raw_apps.split(",") if a.isdigit()]
        lat, lon = float(lat), float(lon)
        ua = appsinstalled_pb2.UserApps()
        ua.lat = lat
        ua.lon = lon
        ua.apps.extend(apps)
        packed = ua.SerializeToString()
        unpacked = appsinstalled_pb2.UserApps()
        unpacked.ParseFromString(packed)
        assert ua == unpacked


if __name__ == '__main__':
    op = OptionParser()
    op.add_option("-t", "--test", action="store_true", default=False)
    op.add_option("-l", "--log", action="store", default=None)
    op.add_option("--dry", action="store_true", default=False)
    op.add_option("--pattern", action="store", default="/data/appsinstalled/*.tsv.gz")
    op.add_option("--idfa", action="store", default="127.0.0.1:33013")
    op.add_option("--gaid", action="store", default="127.0.0.1:33014")
    op.add_option("--adid", action="store", default="127.0.0.1:33015")
    op.add_option("--dvid", action="store", default="127.0.0.1:33016")
    (opts, args) = op.parse_args()
    logging.basicConfig(filename=opts.log, level=logging.INFO if not opts.dry else logging.DEBUG,
                        format='[%(asctime)s] %(levelname).1s %(message)s', datefmt='%Y.%m.%d %H:%M:%S')
    if opts.test:
        prototest()
        sys.exit(0)

    logging.info("Memc loader started with options: %s" % opts)
    start_time = time.time()
    try:
        main(opts)
    except Exception as e:
        logging.exception("Unexpected error: %s" % e)
        sys.exit(1)
    finally:
        logging.info("--- %s seconds ---" % round((time.time() - start_time)))
