{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  # https://devenv.sh/packages/
  packages = [
    # pkgs.gcc
    pkgs.gnum4
    # pkgs.gmp
    # pkgs.mpfr
    pkgs.maturin
    # pkgs.gnumake
    # pkgs.diffutils
    # pkgs.pyright
  ];

  languages.nix = {
    enable = true;
  };

  languages.rust = {
    enable = true;
    channel = "stable";

    components = ["rustc" "cargo" "clippy" "rustfmt" "rust-analyzer"];
  };

  languages.python = {
    enable = true;
    uv.enable = true;
    uv.sync.enable = true;
  };
  languages.cplusplus.enable = true;
}
