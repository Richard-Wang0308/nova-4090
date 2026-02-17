# Step 1: Install PostgreSQL
apt-get update
apt-get install -y postgresql postgresql-contrib

# Step 2: Start PostgreSQL Service
service postgresql start

# Step 3: Create Database & User
su - postgres -c "createdb P31652_rxn1_db"
su - postgres -c "psql -c \"CREATE USER gentle WITH PASSWORD 'gentleman';\""
su - postgres -c "psql -c \"ALTER ROLE gentle SET client_encoding TO 'utf8';\""
su - postgres -c 'psql -c "GRANT ALL PRIVILEGES ON DATABASE \"P31652_rxn1_db\" TO gentle;"'



#DB Interface
    # Add pgAdmin repository
    curl https://www.pgadmin.org/static/packages_pgadmin_org.pub | sudo apt-key add -
    sudo sh -c 'echo "deb https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list'

    # Update and install
    sudo apt update
    sudo apt install pgadmin4


    # Start the service
    sudo systemctl start pgadmin4

    # Enable on boot (optional)
    sudo systemctl enable pgadmin4


#php db interface
apt update
apt install php php-pgsql


cd ~/workspace/test
wget https://github.com/vrana/adminer/releases/download/v4.8.1/adminer-4.8.1-en.php
php -S localhost:8080 adminer-4.8.1-en.php