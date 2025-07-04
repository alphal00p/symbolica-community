{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  # https://devenv.sh/packages/
  packages = [
    pkgs.gnum4
    pkgs.gmp
    pkgs.mpfr
    pkgs.python3
    pkgs.maturin
    pkgs.gnumake
    pkgs.diffutils
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
  };
  languages.cplusplus.enable = true;
}
