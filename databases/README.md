# create databases for testing

## issues

- mysql dbs only accessible with `root` user, even though non-root user is created from environment vars
- not really an "issue" but the postgres db is named after the USER

## prep db sql files

### mysql

need to add these lines before the first table creation:

```
CREATE DATABASE IF NOT EXISTS `<db_name>`;
USE `<db_name>`;
```

### postgres

set the user in the sql and/or docker compose file accordingly by adjusting the username on these lines:

```
ALTER TABLE database.table OWNER TO someusername;
```

## setup

1. create a `db.env` with the following keys, setting the password as you like:

```
MYSQL_ROOT_PASSWORD=<mysql root password>
MYSQL_PASSWORD=<mysql password>
POSTGRES_PASSWORD=<postgres password>
```

2. copy the `*.sql` files into the "backups" directories. max one for postgres.
3. for postgres, add the following to the docker compose file:

```
./postgres/backups/<my_postgresql_file>.sql:/docker-entrypoint-initdb.d/init.sql
```

## starting

use `docker compose up -d` to start. you may need to run it twice due to initialization issues.

ex: `docker compose up -d && docker logs -f testbed_postgres` will start it and show the postgres logs

use `docker ps` to check the status. you should have `testbed_postgres` and `testbed_mysql`

if one or both are missing, try running `docker compose up -d` again.

you can check the logs with `docker logs -f testbed_postgres` and `docker logs -f testbed_mysql`

### successful start

successful start:

```
(base) derek@jennifer:~$ docker ps
CONTAINER ID   IMAGE                COMMAND                  CREATED          STATUS          PORTS                                                                                      NAMES
c3759036d8fe   postgres:14-alpine   "docker-entrypoint.s…"   21 seconds ago   Up 20 seconds   0.0.0.0:5432->5432/tcp, :::5432->5432/tcp                                                  testbed_postgres
f76a7d463769   mysql:8.0            "docker-entrypoint.s…"   21 seconds ago   Up 8 seconds    0.0.0.0:3306->3306/tcp, :::3306->3306/tcp, 0.0.0.0:33060->33060/tcp, :::33060->33060/tcp   testbed_mysql
```

mysql ready logs:

```
2024-11-11T09:13:01.208069Z 0 [System] [MY-013602] [Server] Channel mysql_main configured to support TLS. Encrypted connections are now supported for this channel.
2024-11-11T09:13:01.211417Z 0 [Warning] [MY-011810] [Server] Insecure configuration for --pid-file: Location '/var/run/mysqld' in the path is accessible to all OS users. Consider choosing a different directory.
2024-11-11T09:13:01.219159Z 0 [System] [MY-011323] [Server] X Plugin ready for connections. Bind-address: '::' port: 33060, socket: /var/run/mysqld/mysqlx.sock
2024-11-11T09:13:01.219184Z 0 [System] [MY-010931] [Server] /usr/sbin/mysqld: ready for connections. Version: '8.0.39'  socket: '/var/run/mysqld/mysqld.sock'  port: 3306  MySQL Community Server - GPL
```

postgres ready logs:

```
2024-11-11 09:12:50.257 UTC [1] LOG:  starting PostgreSQL 14.13 on x86_64-pc-linux-musl, compiled by gcc (Alpine 13.2.1_git20240309) 13.2.1 20240309, 64-bit
2024-11-11 09:12:50.257 UTC [1] LOG:  listening on IPv4 address "0.0.0.0", port 5432
2024-11-11 09:12:50.257 UTC [1] LOG:  listening on IPv6 address "::", port 5432
2024-11-11 09:12:50.261 UTC [1] LOG:  listening on Unix socket "/var/run/postgresql/.s.PGSQL.5432"
2024-11-11 09:12:50.269 UTC [59] LOG:  database system was shut down at 2024-11-11 09:12:50 UTC
2024-11-11 09:12:50.274 UTC [1] LOG:  database system is ready to accept connections
```