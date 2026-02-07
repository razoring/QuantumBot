create table Account(
	id bigserial primary key,
	email text unique ,
	phone text,
	discord text unique,
	premium boolean default false,
	preferences jsonb,
	credits bigint default 0,
	created text,
	updated text
);

create table Sensitives(
	id bigserial primary key,
	account bigint references Account(id) unique not null,
	country text,
	birthday text,
	gender text,
	race text,
	updated text
);

create table Alert (
	id bigserial primary key,
	account bigint references Account(id) not null,
	symbol text,
	price double precision,
	updated text
);

create table Ticker(
	id bigserial primary key,
	ticker text unique,
	sector text,
	industry text,
	active boolean default true,
	accuracy double precision,
	weight jsonb,
	datapoints jsonb,
	updated text
);

create table Request (
	id bigserial primary key,
	ticker bigint references Ticker(id) not null, 
	account bigint references Account(id) not null,
	updated text
);

create index RequestAnalytics on Request(account, ticker);
create index AccountLogins on Account(discord, email);
create index AlertsAccount on Alert(account);
create index TickerActive on Ticker(ticker) where active = true;


